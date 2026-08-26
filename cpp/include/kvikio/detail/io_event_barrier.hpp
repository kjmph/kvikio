/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <kvikio/detail/event.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

/**
 * @brief Per-pread barrier that lets the caller's thread wait for every reactor's H2D to finish.
 *
 * Constructed once per device-path `RemoteHandle::pread` and shared via `std::shared_ptr` with all
 * of that pread's sub-range transfers. Each reactor I/O thread records into its own slot (keyed by
 * thread id) after every `cuMemcpyAsync`, re-recording the same event on later calls. Once all
 * sub-ranges report completion, the caller calls `sync_all_events()` to block until each thread's
 * last H2D has drained.
 */
class IoEventBarrier {
 public:
  /**
   * @brief Construct a barrier carrying `cuda_context` as metadata.
   *
   * @param cuda_context The CUDA context that pread's H2Ds will land in. The normal event path is
   * context-agnostic; the barrier pushes this context only for its last-resort context-wide fence.
   */
  explicit IoEventBarrier(CUcontext cuda_context) noexcept;

  IoEventBarrier(IoEventBarrier const&)            = delete;
  IoEventBarrier& operator=(IoEventBarrier const&) = delete;
  IoEventBarrier(IoEventBarrier&&)                 = delete;
  IoEventBarrier& operator=(IoEventBarrier&&)      = delete;

  ~IoEventBarrier() noexcept = default;

  /**
   * @brief Get the CUDA context this barrier was constructed with.
   *
   * @return The stored `CUcontext`.
   */
  [[nodiscard]] CUcontext cuda_context() const noexcept;

  /**
   * @brief Record an event on `stream` in the calling thread's slot, creating the slot on first
   * use.
   *
   * @param stream The CUDA stream to record on. Must belong to the same context as the event, i.e.
   * the context current at first record.
   * @exception kvikio::CUfileException if the underlying `cuEventRecord` fails.
   */
  void record_event(CUstream stream);

  /**
   * @brief Record that an H2D may have been submitted without a usable completion event.
   *
   * A reactor calls this only when both event recording and its local stream-synchronization
   * recovery fail. The caller-side barrier will then use a context-wide fence before resolving the
   * I/O future.
   */
  void mark_completion_unknown() noexcept;

  /**
   * @brief Block the calling thread until every recorded event has signaled.
   *
   * Every recorded event is attempted even if an earlier event wait fails. If an event wait fails,
   * or a reactor reported an unrecorded H2D, this method performs a last-resort context-wide fence.
   * A successful context fence proves all preceding H2Ds are quiescent, after which the first event
   * error is rethrown. If the context fence itself fails, the process is terminated: returning to
   * the caller would permit destruction of a destination that CUDA may still be writing.
   *
   * This method is context-agnostic in that the CUDA context current on the calling thread may
   * differ from the context used by the H2Ds.
   *
   * @exception kvikio::CUfileException if an event wait fails after the context fallback succeeds.
   */
  void sync_all_events();

 private:
  CUcontext _cuda_context;
  std::atomic<bool> _completion_unknown{false};
  std::mutex _mutex;
  std::unordered_map<std::thread::id, CudaEventPool::CudaEvent> _thread_events;
};

/**
 * @brief Resolve an asynchronous I/O result only after all associated CUDA work is quiescent.
 *
 * The event barrier is synchronized even when `completion.get()` throws. This preserves the
 * destination-buffer lifetime contract on failed multi-part reads: callers may observe the error
 * and release or reuse their device buffer only after every H2D already submitted by a sibling
 * transfer has completed.
 *
 * If an event synchronization fails, a context-wide fence establishes safe destination reuse
 * before the synchronization error is propagated. Failure of that last-resort fence terminates
 * the process because the destination lifetime can no longer be made safe.
 * If both I/O and event synchronization fail, the earlier I/O error remains the primary result
 * after the context-wide fence succeeds; the secondary CUDA error is logged.
 *
 * @param completion The aggregate asynchronous I/O result.
 * @param barrier The CUDA-event barrier shared by the aggregate I/O operation.
 * @return The completed byte count.
 */
std::size_t wait_for_io_completion(std::future<std::size_t> completion, IoEventBarrier& barrier);

/**
 * @brief Construct the deferred device-I/O completion future before any transfer is submitted.
 *
 * The returned future owns both the aggregate network completion and the CUDA event barrier. Its
 * deferred callable reports the byte count only after all submitted H2D work is quiescent,
 * including when the aggregate I/O result is exceptional.
 *
 * This factory can allocate and throw. Callers must invoke it before publishing transfers that can
 * write to caller-owned device memory.
 *
 * @param completion The aggregate asynchronous I/O result.
 * @param barrier The event barrier shared by all device subranges.
 * @return A deferred future that resolves network completion and the CUDA lifetime fence.
 */
std::future<std::size_t> make_io_completion_future(std::future<std::size_t> completion,
                                                   std::shared_ptr<IoEventBarrier> barrier);

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
/**
 * @brief Make the next completion-future factory call throw before constructing its shared state.
 *
 * Test-only entry point used to prove device transfers are not submitted before every potentially
 * throwing completion object exists.
 */
void inject_io_completion_future_failure_for_testing() noexcept;
#endif

}  // namespace kvikio::detail
