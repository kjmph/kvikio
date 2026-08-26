/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <new>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

#include "utils/utils.hpp"

namespace {

CUcontext current_context()
{
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetCurrent(&ctx));
  return ctx;
}

struct StreamGate {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
};

void CUDA_CB wait_at_stream_gate(void* opaque)
{
  auto& gate = *static_cast<StreamGate*>(opaque);
  gate.entered.store(true, std::memory_order_release);
  while (!gate.release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

void CUDA_CB mark_stream_complete(void* opaque)
{
  static_cast<std::atomic<bool>*>(opaque)->store(true, std::memory_order_release);
}

CUresult CUDAAPI fail_launch_host_func(CUstream, CUhostFn, void*)
{
  return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI fail_stream_synchronize(CUstream) { return CUDA_ERROR_INVALID_VALUE; }

class ScopedLaunchHostFuncOverride {
 public:
  using Function = decltype(kvikio::cudaAPI::instance().LaunchHostFunc);

  explicit ScopedLaunchHostFuncOverride(Function replacement)
    : _slot{kvikio::cudaAPI::instance().LaunchHostFunc}, _saved{_slot}
  {
    _slot = replacement;
  }

  ~ScopedLaunchHostFuncOverride() { _slot = _saved; }

 private:
  Function& _slot;
  Function _saved;
};

class ScopedStreamSynchronizeOverride {
 public:
  using Function = decltype(kvikio::cudaAPI::instance().StreamSynchronize);

  explicit ScopedStreamSynchronizeOverride(Function replacement)
    : _slot{kvikio::cudaAPI::instance().StreamSynchronize}, _saved{_slot}
  {
    _slot = replacement;
  }

  ~ScopedStreamSynchronizeOverride() { _slot = _saved; }

 private:
  Function& _slot;
  Function _saved;
};

}  // namespace

class BounceBufferCacheTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
};

TEST_F(BounceBufferCacheTest, try_get_returns_buffer_under_cap)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(4);
  EXPECT_EQ(cache.cap(), std::optional<std::size_t>{4});

  auto ctx = current_context();
  auto b   = cache.try_get(ctx);
  EXPECT_TRUE(b.has_value());
  EXPECT_NE(b->get(), nullptr);
  EXPECT_EQ(b->size(), kvikio::defaults::bounce_buffer_size());
}

TEST_F(BounceBufferCacheTest, try_get_returns_nullopt_at_cap)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);

  auto ctx = current_context();
  auto b1  = cache.try_get(ctx);
  auto b2  = cache.try_get(ctx);
  EXPECT_TRUE(b1.has_value());
  EXPECT_TRUE(b2.has_value());

  // Cap of 2 is reached: third try_get must fail.
  auto b3 = cache.try_get(ctx);
  EXPECT_FALSE(b3.has_value());
}

TEST_F(BounceBufferCacheTest, cap_nullopt_means_unlimited)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(
    std::nullopt);
  EXPECT_FALSE(cache.cap().has_value());

  auto ctx = current_context();
  std::vector<decltype(cache.try_get(ctx))> bufs;
  for (int i = 0; i < 32; ++i) {
    auto b = cache.try_get(ctx);
    EXPECT_TRUE(b.has_value()) << "iteration " << i;
    bufs.push_back(std::move(b));
  }
}

TEST_F(BounceBufferCacheTest, recycle_now_returns_buffer_to_free_list)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  void* first_ptr = nullptr;
  {
    auto b = cache.try_get(ctx);
    EXPECT_TRUE(b.has_value());
    first_ptr = b->get();
    cache.recycle_now(ctx, std::move(*b));
  }

  // The recycled buffer should come back as the next try_get (LIFO via the free list).
  auto b2 = cache.try_get(ctx);
  EXPECT_TRUE(b2.has_value());
  EXPECT_EQ(b2->get(), first_ptr);
}

TEST_F(BounceBufferCacheTest, recycle_after_round_trip)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  void* first_ptr = nullptr;
  {
    auto b = cache.try_get(ctx);
    EXPECT_TRUE(b.has_value());
    first_ptr = b->get();
    cache.recycle_after(ctx, std::move(*b), stream);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  // After sync, the callback has run and the buffer is on the free list.
  auto b2 = cache.try_get(ctx);
  EXPECT_TRUE(b2.has_value());
  EXPECT_EQ(b2->get(), first_ptr);
  cache.recycle_now(ctx, std::move(*b2));

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, recycle_callback_submission_failure_preserves_slot_accounting)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(1);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  auto buffer = cache.try_get(ctx);
  ASSERT_TRUE(buffer.has_value());

  {
    ScopedLaunchHostFuncOverride const override{&fail_launch_host_func};
    EXPECT_THROW(cache.recycle_after(ctx, std::move(*buffer), stream), std::exception);
  }

  // Launch failure drains the stream and removes the slot from this shard. The outer recovery sees
  // that ownership already moved and must not decrement checked_out a second time.
  auto replacement = cache.try_get(ctx);
  EXPECT_TRUE(replacement.has_value());
  if (replacement.has_value()) { cache.recycle_now(ctx, std::move(*replacement)); }
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, accounting_failure_leaves_source_owned_for_synchronized_recovery)
{
  using Cache = kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator>;
  Cache cache(1);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  auto buffer = cache.try_get(ctx);
  ASSERT_TRUE(buffer.has_value());
  auto* const original = buffer->get();

  std::atomic<bool> preceding_work_completed{false};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().LaunchHostFunc(
    stream, &mark_stream_complete, &preceding_work_completed));
  kvikio::detail::inject_bounce_buffer_cache_failure_for_testing(
    kvikio::detail::BounceBufferCacheFailurePoint::ACCOUNTING_TRANSITION);
  EXPECT_THROW(cache.recycle_after(ctx, std::move(*buffer), stream), std::bad_alloc);

  // The recycle path must synchronize preceding stream work and recover the still-owned buffer
  // after callback-state accounting fails.
  EXPECT_TRUE(preceding_work_completed.load(std::memory_order_acquire));
  auto recycled = cache.try_get(ctx);
  ASSERT_TRUE(recycled.has_value());
  EXPECT_EQ(recycled->get(), original);
  cache.recycle_now(ctx, std::move(*recycled));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, callback_insertion_failure_detaches_buffers_and_poisons_cache)
{
  using Cache = kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator>;
  Cache cache(std::nullopt);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  auto buffer = cache.try_get(ctx);
  ASSERT_TRUE(buffer.has_value());
  auto* const allocation      = buffer->get();
  auto const allocation_size  = buffer->size();
  auto& underlying_pool       = kvikio::CudaPinnedBounceBufferPool::instance();
  auto const pool_free_before = underlying_pool.num_free_buffers();

  kvikio::detail::inject_bounce_buffer_cache_failure_for_testing(
    kvikio::detail::BounceBufferCacheFailurePoint::CALLBACK_INSERTION);
  cache.recycle_after(ctx, std::move(*buffer), stream);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  // A CUDA callback cannot safely return a pinned allocation through the global pool because its
  // exceptional paths can invoke CUDA. The injected insertion failure must detach it and fail the
  // cache closed instead.
  EXPECT_THROW(std::ignore = cache.try_get(ctx), std::runtime_error);
  ASSERT_EQ(underlying_pool.num_free_buffers(), pool_free_before);

  // Failure injection is synchronized, so the test thread can reclaim the deliberately detached
  // allocation without issuing a CUDA API from the callback thread.
  kvikio::CudaPinnedAllocator{}.deallocate(allocation, allocation_size);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, failed_callback_and_sync_poison_cache_without_replacement_allocation)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(1);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  auto buffer = cache.try_get(ctx);
  ASSERT_TRUE(buffer.has_value());

  {
    ScopedLaunchHostFuncOverride const launch_override{&fail_launch_host_func};
    ScopedStreamSynchronizeOverride const sync_override{&fail_stream_synchronize};
    EXPECT_THROW(cache.recycle_after(ctx, std::move(*buffer), stream), std::exception);
  }

  // CUDA could not prove the leaked source slot was quiescent. The shard must fail closed instead
  // of dropping its in-flight count and allocating an unbounded sequence of replacements.
  EXPECT_THROW(std::ignore = cache.try_get(ctx), std::runtime_error);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, recycle_after_releases_in_flight_slot)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  // Fill the cap, schedule both to be recycled, then verify try_get eventually succeeds
  // after the callbacks fire.
  {
    auto b1 = cache.try_get(ctx);
    auto b2 = cache.try_get(ctx);
    EXPECT_TRUE(b1.has_value());
    EXPECT_TRUE(b2.has_value());
    EXPECT_FALSE(cache.try_get(ctx).has_value());  // at cap
    cache.recycle_after(ctx, std::move(*b1), stream);
    cache.recycle_after(ctx, std::move(*b2), stream);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  // After both callbacks have run, the cache is back to fully free.
  auto b = cache.try_get(ctx);
  EXPECT_TRUE(b.has_value());
  cache.recycle_now(ctx, std::move(*b));

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, recycle_after_retains_source_until_stream_reaches_callback)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(1);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  auto buffer = cache.try_get(ctx);
  ASSERT_TRUE(buffer.has_value());
  auto* const original = buffer->get();

  StreamGate gate;
  KVIKIO_CUDA_DRIVER_TRY(
    kvikio::cudaAPI::instance().LaunchHostFunc(stream, &wait_at_stream_gate, &gate));
  cache.recycle_after(ctx, std::move(*buffer), stream);

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!gate.entered.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  if (!gate.entered.load(std::memory_order_acquire)) {
    gate.release.store(true, std::memory_order_release);
    std::ignore = kvikio::cudaAPI::instance().StreamSynchronize(stream);
    std::ignore = kvikio::cudaAPI::instance().StreamDestroy(stream);
    FAIL() << "CUDA stream did not reach the lifetime gate";
  }

  // The cached pinned buffer remains unavailable while the preceding H2D work is still in flight.
  EXPECT_FALSE(cache.try_get(ctx).has_value());
  gate.release.store(true, std::memory_order_release);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  auto recycled = cache.try_get(ctx);
  ASSERT_TRUE(recycled.has_value());
  EXPECT_EQ(recycled->get(), original);
  cache.recycle_now(ctx, std::move(*recycled));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, multi_context_isolation)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);

  auto primary_ctx = current_context();
  EXPECT_NE(primary_ctx, nullptr);

  // Fill the primary-context cap.
  auto b1 = cache.try_get(primary_ctx);
  auto b2 = cache.try_get(primary_ctx);
  EXPECT_TRUE(b1.has_value());
  EXPECT_TRUE(b2.has_value());
  EXPECT_FALSE(cache.try_get(primary_ctx).has_value());

  // Create a second context on the same device. The per-key cap is per (thread, ctx), so the second
  // context has its own independent budget.
  CUdevice dev{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetDevice(&dev));
  CUcontext second_ctx{};
#if CUDA_VERSION >= 13000
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, nullptr, 0, dev));
#else
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, 0, dev));
#endif
  EXPECT_EQ(current_context(), second_ctx);

  auto s1 = cache.try_get(second_ctx);
  auto s2 = cache.try_get(second_ctx);
  EXPECT_TRUE(s1.has_value());
  EXPECT_TRUE(s2.has_value());
  EXPECT_FALSE(cache.try_get(second_ctx).has_value());

  cache.recycle_now(second_ctx, std::move(*s1));
  cache.recycle_now(second_ctx, std::move(*s2));

  // Restore the primary context and clean up.
  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  EXPECT_EQ(popped, second_ctx);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxDestroy(second_ctx));
  EXPECT_EQ(current_context(), primary_ctx);

  cache.recycle_now(primary_ctx, std::move(*b1));
  cache.recycle_now(primary_ctx, std::move(*b2));
}

TEST_F(BounceBufferCacheTest, per_thread_isolation)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(1);
  auto ctx = current_context();

  // Main thread occupies its key's single slot.
  auto b1 = cache.try_get(ctx);
  EXPECT_TRUE(b1.has_value());
  EXPECT_FALSE(cache.try_get(ctx).has_value());

  // A worker thread has an independent (thread, ctx) key and should not be blocked.
  std::atomic<bool> worker_succeeded{false};
  std::thread worker([&] {
    KVIKIO_CHECK_CUDA(cudaSetDevice(0));
    auto wb = cache.try_get(ctx);
    if (wb.has_value()) {
      worker_succeeded = true;
      cache.recycle_now(ctx, std::move(*wb));
    }
  });
  worker.join();
  EXPECT_TRUE(worker_succeeded.load());

  cache.recycle_now(ctx, std::move(*b1));
}

TEST_F(BounceBufferCacheTest, concurrent_get_and_recycle_now)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(
    std::nullopt);  // unlimited

  constexpr int num_threads           = 8;
  constexpr int iterations_per_thread = 64;
  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&] {
      try {
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        auto ctx = current_context();
        for (int i = 0; i < iterations_per_thread; ++i) {
          auto b = cache.try_get(ctx);
          if (!b.has_value()) {
            ++errors;
            continue;
          }
          cache.recycle_now(ctx, std::move(*b));
        }
      } catch (...) {
        ++errors;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_EQ(errors.load(), 0);
}

TEST_F(BounceBufferCacheTest, singleton_instance_has_default_cap)
{
  auto& s =
    kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator>::instance();
  auto const max_total    = kvikio::defaults::remote_io_max_concurrent_requests();
  auto const n            = kvikio::defaults::remote_io_num_reactors();
  auto const expected_cap = kvikio::detail::bounce_buffer_cache_shard_limit(max_total, n);
  EXPECT_EQ(s.cap(), expected_cap);

  // try_get on the singleton works.
  auto b = s.try_get(current_context());
  EXPECT_TRUE(b.has_value());
  EXPECT_NE(b->get(), nullptr);
  s.recycle_now(current_context(), std::move(*b));
}

TEST(BounceBufferCacheShardLimitTest, unlimited)
{
  EXPECT_FALSE(kvikio::detail::bounce_buffer_cache_shard_limit(0, 16).has_value());
}

TEST(BounceBufferCacheShardLimitTest, rounds_up_to_busiest_reactor)
{
  EXPECT_EQ(kvikio::detail::bounce_buffer_cache_shard_limit(16, 16), 1);
  EXPECT_EQ(kvikio::detail::bounce_buffer_cache_shard_limit(192, 16), 12);
  EXPECT_EQ(kvikio::detail::bounce_buffer_cache_shard_limit(193, 16), 13);
  EXPECT_EQ(kvikio::detail::bounce_buffer_cache_shard_limit(195, 16), 13);
}

TEST(BounceBufferCacheShardLimitTest, rejects_invalid_configuration)
{
  EXPECT_THROW((void)kvikio::detail::bounce_buffer_cache_shard_limit(0, 0), std::invalid_argument);
  EXPECT_THROW((void)kvikio::detail::bounce_buffer_cache_shard_limit(16, 0), std::invalid_argument);
  EXPECT_THROW((void)kvikio::detail::bounce_buffer_cache_shard_limit(1, 16), std::invalid_argument);
}
