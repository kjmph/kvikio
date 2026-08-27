/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <kvikio/detail/event.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

namespace {

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
std::atomic<bool> fail_next_completion_future{false};
#endif

[[noreturn]] void terminate_after_failed_context_fence() noexcept
{
  try {
    KVIKIO_LOG_ERROR(
      "context-wide CUDA fence failed; terminating to preserve device-buffer lifetime");
  } catch (...) {
    // Logging is best effort on a fatal path and must never bypass the fail-stop policy.
  }
  std::terminate();
}

}  // namespace

IoEventBarrier::IoEventBarrier(CUcontext cuda_context) noexcept : _cuda_context{cuda_context} {}

CUcontext IoEventBarrier::cuda_context() const noexcept { return _cuda_context; }

void IoEventBarrier::prepare_event(CUstream stream)
{
  auto const key = EventKey{std::this_thread::get_id(), stream};
  {
    std::lock_guard const lock(_mutex);
    if (auto it = _events.find(key); it != _events.end()) {
      if (it->second.get() != nullptr) { return; }
      _events.erase(it);
    }
  }

  auto event = CudaEventPool::instance().get();
  std::lock_guard const lock(_mutex);
  // Retain the second check because another caller can prepare the same pair while CUDA creates the
  // event outside the barrier mutex.
  if (auto it = _events.find(key); it != _events.end()) {
    if (it->second.get() != nullptr) { return; }
    _events.erase(it);
  }
  auto [_, inserted] = _events.emplace(key, std::move(event));
  KVIKIO_EXPECT(inserted, "New event insertion failed unexpectedly.");
}

void IoEventBarrier::record_prepared_event(CUstream stream)
{
  CudaEventPool::CudaEvent* event_ptr{nullptr};
  auto const key = EventKey{std::this_thread::get_id(), stream};
  {
    std::lock_guard const lock(_mutex);
    auto const it = _events.find(key);
    KVIKIO_EXPECT(it != _events.end() && it->second.get() != nullptr,
                  "No CUDA event was prepared for this thread and stream.",
                  std::logic_error);
    event_ptr = &it->second;
  }

  // Note that for the node-based unordered_map, pointers (or references) to either key or data
  // stored in the container can never be invalidated by insertion, even when the corresponding
  // iterator is invalidated. So it is safe to move this function outside the mutex.
  try {
    event_ptr->record(stream);
  } catch (...) {
    auto const record_error = std::current_exception();
    mark_completion_unknown();
    // Do not return an event that failed to record to the process-wide reuse pool.
    event_ptr->abandon();
    try {
      std::lock_guard const lock(_mutex);
      auto it = _events.find(key);
      if (it != _events.end() && &it->second == event_ptr) { _events.erase(it); }
    } catch (...) {
      // The abandoned map entry owns no CUDA handle and can be removed on the next record. Preserve
      // the event-record failure as the primary error.
    }
    std::rethrow_exception(record_error);
  }
}

void IoEventBarrier::record_event(CUstream stream)
{
  prepare_event(stream);
  record_prepared_event(stream);
}

void IoEventBarrier::mark_completion_unknown() noexcept
{
  _completion_unknown.store(true, std::memory_order_release);
}

void IoEventBarrier::sync_all_events()
{
  std::exception_ptr first_event_error;
  bool needs_context_fence = _completion_unknown.load(std::memory_order_acquire);
  try {
    {
      // Iterating in place avoids an allocation failure after I/O that would otherwise bypass
      // every fence. Aggregate completion precedes this call, so the lock is uncontended in normal
      // operation and reactors can no longer add records.
      std::lock_guard const lock(_mutex);
      for (auto& [key, event] : _events) {
        try {
          event.synchronize();
        } catch (...) {
          if (first_event_error == nullptr) { first_event_error = std::current_exception(); }
          needs_context_fence = true;
          // A failed event must never be handed to a later operation through the global pool.
          event.abandon();
        }
      }
    }
  } catch (...) {
    // Even mutex acquisition is a potentially throwing operation. Preserve that error and use the
    // allocation-free context fence rather than returning with all event state unvisited.
    if (first_event_error == nullptr) { first_event_error = std::current_exception(); }
    needs_context_fence = true;
  }

  if (needs_context_fence) {
    // CUDA documents cuCtxSynchronize as blocking until all preceding work has completed. It is
    // the only available fence for an H2D whose event could not be recorded or waited. If this
    // fallback fails, returning would let the caller release a destination that may still be in
    // use, so fail-stop is the only memory-safe outcome.
    try {
      PushAndPopContext current{_cuda_context};
      KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxSynchronize());
    } catch (...) {
      terminate_after_failed_context_fence();
    }
  }

  if (first_event_error != nullptr) { std::rethrow_exception(first_event_error); }
}

std::size_t wait_for_io_completion(std::future<std::size_t> completion, IoEventBarrier& barrier)
{
  std::size_t result{};
  std::exception_ptr io_error;
  try {
    result = completion.get();
  } catch (...) {
    io_error = std::current_exception();
  }

  // This must run before the I/O error escapes. Other subranges may already have queued copies
  // into the caller-owned destination even when one subrange fails.
  std::exception_ptr completion_error;
  try {
    barrier.sync_all_events();
  } catch (...) {
    completion_error = std::current_exception();
  }

  if (io_error != nullptr && completion_error != nullptr) {
    // The I/O failure happened first and remains the operation's primary error. The successful
    // context fallback inside sync_all_events() has already made destination reuse safe.
    try {
      std::rethrow_exception(completion_error);
    } catch (std::exception const& error) {
      try {
        KVIKIO_LOG_ERROR(std::string{"secondary CUDA completion failure after I/O error: "} +
                         error.what());
      } catch (...) {
      }
    } catch (...) {
      try {
        KVIKIO_LOG_ERROR("secondary unknown CUDA completion failure after I/O error");
      } catch (...) {
      }
    }
  }

  if (io_error != nullptr) { std::rethrow_exception(io_error); }
  if (completion_error != nullptr) { std::rethrow_exception(completion_error); }
  return result;
}

std::future<std::size_t> make_io_completion_future(std::future<std::size_t> completion,
                                                   std::shared_ptr<IoEventBarrier> barrier)
{
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  if (fail_next_completion_future.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc{};
  }
#endif
  KVIKIO_EXPECT(barrier != nullptr,
                "device I/O completion requires a non-null event barrier",
                std::invalid_argument);
  return std::async(std::launch::deferred,
                    [completion = std::move(completion), barrier = std::move(barrier)]() mutable {
                      return wait_for_io_completion(std::move(completion), *barrier);
                    });
}

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
void inject_io_completion_future_failure_for_testing() noexcept
{
  fail_next_completion_future.store(true, std::memory_order_relaxed);
}
#endif

}  // namespace kvikio::detail
