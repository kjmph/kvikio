/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

namespace {

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
constexpr int failure_injection_disabled = -1;
std::atomic<int> recycle_failure_point{failure_injection_disabled};

void maybe_inject_recycle_failure(BounceBufferCacheFailurePoint point)
{
  auto expected = static_cast<int>(point);
  if (recycle_failure_point.compare_exchange_strong(expected,
                                                    failure_injection_disabled,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_acquire)) {
    throw std::bad_alloc{};
  }
}
#endif

void log_failure_noexcept(char const* prefix, std::exception_ptr failure) noexcept
{
  try {
    if (failure != nullptr) { std::rethrow_exception(failure); }
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + "unknown exception");
    } catch (...) {
    }
  } catch (std::exception const& error) {
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + error.what());
    } catch (...) {
    }
  } catch (...) {
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + "unknown exception");
    } catch (...) {
    }
  }
}

}  // namespace

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
void inject_bounce_buffer_cache_failure_for_testing(BounceBufferCacheFailurePoint point) noexcept
{
  recycle_failure_point.store(static_cast<int>(point), std::memory_order_release);
}
#endif

std::optional<std::size_t> bounce_buffer_cache_shard_limit(std::size_t max_total,
                                                           std::size_t num_reactors)
{
  KVIKIO_EXPECT(
    num_reactors > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);
  if (max_total == 0) { return std::nullopt; }
  KVIKIO_EXPECT(max_total >= num_reactors,
                "remote_io_max_concurrent_requests must be zero (unlimited) or at least "
                "remote_io_num_reactors",
                std::invalid_argument);

  return max_total / num_reactors + static_cast<std::size_t>(max_total % num_reactors != 0);
}

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>::Shard::Shard(std::optional<std::size_t> cap)
{
  if (cap.has_value()) { free.reserve(cap.value()); }
}

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>::BounceBufferCachePerThreadAndContext(
  std::optional<std::size_t> cap)
  : _cap{cap}
{
}

template <typename Allocator>
std::optional<std::size_t> BounceBufferCachePerThreadAndContext<Allocator>::cap() const noexcept
{
  return _cap;
}

template <typename Allocator>
typename BounceBufferCachePerThreadAndContext<Allocator>::Shard&
BounceBufferCachePerThreadAndContext<Allocator>::get_shard(CUcontext ctx)
{
  auto const key = std::pair{std::this_thread::get_id(), ctx};
  std::lock_guard const lock(_map_mutex);
  auto it = _shards.find(key);
  if (it == _shards.end()) { it = _shards.emplace(key, std::make_unique<Shard>(cap())).first; }
  return *it->second;
}

template <typename Allocator>
std::optional<typename BounceBufferCachePerThreadAndContext<Allocator>::Buffer>
BounceBufferCachePerThreadAndContext<Allocator>::try_get(CUcontext ctx)
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(!_emergency_poisoned.load(std::memory_order_acquire),
                "bounce-buffer cache is poisoned after CUDA could not establish safe buffer reuse",
                std::runtime_error);
  auto& shard = get_shard(ctx);

  std::vector<Buffer> stale_buffers;
  std::optional<Buffer> reused_buffer;
  bool cap_reached{false};
  {
    std::lock_guard const lock(shard.mutex);

    KVIKIO_EXPECT(!shard.poisoned,
                  "bounce-buffer cache is poisoned after CUDA could not establish safe buffer "
                  "reuse",
                  std::runtime_error);

    // Discard free buffers whose size no longer matches the current bounce_buffer_size. Their
    // destructors route through BounceBufferPool::put, which deallocates wrong-size buffers.
    auto const current_size = defaults::bounce_buffer_size();
    while (!shard.free.empty() && shard.free.back().size() != current_size) {
      stale_buffers.push_back(std::move(shard.free.back()));
      shard.free.pop_back();
    }

    if (!shard.free.empty()) {
      reused_buffer = std::move(shard.free.back());
      shard.free.pop_back();
      ++shard.checked_out;
    } else {
      // No buffer available on the free list. Allocate if under cap (or if cap is unlimited).
      auto const total = shard.free.size() + shard.checked_out + shard.in_flight;
      if (_cap.has_value() && total >= _cap.value()) {
        cap_reached = true;
      } else {
        // reused_buffer does not contain a value (std::nullopt)
        ++shard.checked_out;
      }
    }
  }

  stale_buffers.clear();

  if (reused_buffer.has_value() || cap_reached) { return reused_buffer; }

  try {
    return BounceBufferPool<Allocator>::instance().get();
  } catch (...) {
    std::lock_guard const lock(shard.mutex);
    --shard.checked_out;
    throw;
  }
}

template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::abandon_checked_out_after_failed_sync(
  CUcontext ctx, Buffer&& buf) noexcept
{
  if (buf.get() == nullptr) { return; }
  // Detach the allocation before any lock or logging operation can fail. CUDA could not prove that
  // asynchronous readers stopped touching it, so its destructor must never run.
  std::ignore = buf.release();
  try {
    auto& shard = get_shard(ctx);
    std::lock_guard const lock(shard.mutex);
    shard.poisoned = true;
  } catch (...) {
    auto const poison_error = std::current_exception();
    _emergency_poisoned.store(true, std::memory_order_release);
    log_failure_noexcept("failed to poison bounce-buffer cache shard: ", poison_error);
  }
}

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::erase_poisoned_shard_for_testing(
  CUcontext ctx)
{
  auto const key = std::pair{std::this_thread::get_id(), ctx};
  std::lock_guard const map_lock(_map_mutex);
  auto const it = _shards.find(key);
  KVIKIO_EXPECT(
    it != _shards.end(), "test cleanup could not find poisoned shard", std::logic_error);
  {
    std::lock_guard const shard_lock(it->second->mutex);
    KVIKIO_EXPECT(
      it->second->poisoned, "test cleanup refuses to erase a healthy shard", std::logic_error);
    KVIKIO_EXPECT(it->second->in_flight == 0 && it->second->free.empty(),
                  "test cleanup refuses to erase a shard that still owns buffers",
                  std::logic_error);
  }
  _shards.erase(it);
}
#endif

template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::recycle_now(CUcontext ctx, Buffer&& buf)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto& shard = get_shard(ctx);
  std::lock_guard const lock(shard.mutex);
  shard.free.push_back(std::move(buf));
  --shard.checked_out;
}

template <typename Allocator>
void BounceBufferCachePerThreadAndContext<Allocator>::recycle_after(
  CUcontext ctx, Buffer&& buf, CUstream stream, std::function<void()> on_recycle)
{
  KVIKIO_NVTX_FUNC_RANGE();
  try {
    auto& shard = get_shard(ctx);
    // Allocate every piece of callback state before taking the caller's buffer. On allocation
    // failure the source remains intact so this function can synchronize and recover it.
    auto data = std::make_unique<RecycleCallbackData>(
      this, &shard, Buffer{nullptr, nullptr, 0}, std::move(on_recycle));

    // Phase A (`checked_out`) ends and Phase B (`in_flight`) starts while the caller still owns the
    // buffer. If the accounting lock or validation fails, recovery can synchronize the intact
    // source.
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
    maybe_inject_recycle_failure(BounceBufferCacheFailurePoint::ACCOUNTING_TRANSITION);
#endif
    {
      std::lock_guard const lock(shard.mutex);
      KVIKIO_EXPECT(shard.checked_out > 0,
                    "recycle_after received a buffer that is not checked out",
                    std::logic_error);
      --shard.checked_out;
      ++shard.in_flight;
    }

    // Buffer move assignment is noexcept. No fallible work occurs between the accounting
    // transition and callback ownership.
    data->buffer = std::move(buf);

    try {
      KVIKIO_CUDA_DRIVER_TRY(
        cudaAPI::instance().LaunchHostFunc(stream, &recycle_callback, data.get()));
    } catch (...) {
      auto const primary_error = std::current_exception();
      // The H2D was enqueued before this callback. If callback submission fails, do not return its
      // source allocation to the pool until the stream has drained.
      bool reuse_is_safe{true};
      try {
        KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      } catch (...) {
        auto const sync_error = std::current_exception();
        reuse_is_safe         = false;
        // CUDA cannot establish that DMA has stopped reading this allocation. Intentionally leak
        // it instead of exposing it to a later network receive.
        std::ignore = data->buffer.release();
        _emergency_poisoned.store(true, std::memory_order_release);
        try {
          std::lock_guard const lock(shard.mutex);
          shard.poisoned = true;
        } catch (...) {
        }
        log_failure_noexcept("recycle_after: stream synchronization failed: ", sync_error);
      }
      // `data` still owns the payload, so its destructor returns the now-drained buffer to the
      // underlying pool. The buffer leaves this shard, so only decrement in_flight. When
      // synchronization also fails, retain the leaked slot in capacity accounting and poison the
      // cache.
      if (reuse_is_safe) {
        try {
          std::lock_guard const lock(shard.mutex);
          --shard.in_flight;
        } catch (...) {
          _emergency_poisoned.store(true, std::memory_order_release);
        }
      }
      std::rethrow_exception(primary_error);
    }

    // The callback owns the heap payload. Here we disown it so this unique_ptr's destructor does
    // not also delete it. If the callback has already run on another thread and freed the payload,
    // `release()` returns a dangling pointer, which we ignore, so that is harmless.
    std::ignore = data.release();
  } catch (...) {
    if (buf.get() != nullptr) {
      // Allocation or accounting failed before ownership transfer. The H2D precedes this call, so
      // drain it before restoring the still-local source to the cache.
      try {
        KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
        recycle_now(ctx, std::move(buf));
      } catch (...) {
        // Recovery could not restore the source without risking reuse. Keep the slot charged and
        // poison the shard before leaking it.
        abandon_checked_out_after_failed_sync(ctx, std::move(buf));
      }
    }
    throw;
  }
}

template <typename Allocator>
void CUDA_CB BounceBufferCachePerThreadAndContext<Allocator>::recycle_callback(void* user_data)
{
  // Runs on a CUDA driver controlled thread. Must not make CUDA API calls. Must be short.
  std::unique_ptr<RecycleCallbackData> data(static_cast<RecycleCallbackData*>(user_data));
  try {
    {
      std::lock_guard const lock(data->shard->mutex);
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
      maybe_inject_recycle_failure(BounceBufferCacheFailurePoint::CALLBACK_INSERTION);
#endif
      data->shard->free.push_back(std::move(data->buffer));
      --data->shard->in_flight;
    }
  } catch (...) {
    auto const failure = std::current_exception();
    // Buffer destruction routes through BounceBufferPool and can invoke CUDA when the configured
    // buffer size changed. Detach the allocation before callback state unwinds, then prevent this
    // cache from handing out a buffer whose recycle accounting is no longer trustworthy.
    std::ignore = data->buffer.release();
    data->owner->_emergency_poisoned.store(true, std::memory_order_release);
    log_failure_noexcept("BounceBufferCachePerThreadAndContext::recycle_callback: ", failure);
    return;
  }

  try {
    if (data->on_recycle) { data->on_recycle(); }
  } catch (...) {
    // Buffer ownership and accounting are already safe; a notification failure must not poison
    // the cache or escape a CUDA callback.
    log_failure_noexcept("BounceBufferCachePerThreadAndContext::recycle_callback notification: ",
                         std::current_exception());
  }
}

template <typename Allocator>
BounceBufferCachePerThreadAndContext<Allocator>&
BounceBufferCachePerThreadAndContext<Allocator>::instance()
{
  KVIKIO_NVTX_FUNC_RANGE();
  static auto* _instance = []() {
    auto const max_total = defaults::remote_io_max_concurrent_requests();
    auto const n         = defaults::remote_io_num_reactors();
    return new BounceBufferCachePerThreadAndContext(bounce_buffer_cache_shard_limit(max_total, n));
  }();
  return *_instance;
}

// Explicit instantiations
template class BounceBufferCachePerThreadAndContext<PageAlignedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
template class BounceBufferCachePerThreadAndContext<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio::detail
