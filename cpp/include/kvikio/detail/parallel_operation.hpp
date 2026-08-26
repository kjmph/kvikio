/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cassert>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/threadpool_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

/**
 * @brief Utility function to create a copyable callable from a move-only callable.
 *
 * The underlying thread pool uses `std::function` (until C++23) or `std::move_only_function`
 * (since C++23) as the element type of the task queue. For the former case that currently applies,
 * the `std::function` requires its "target" (associated callable) to be copy-constructible. This
 * utility function is a workaround for those move-only callables.
 *
 * @tparam F Callable type. F shall be move-only.
 * @param op Callable.
 * @return A new callable that satisfies the copy-constructible condition.
 */
template <typename F>
auto make_copyable_lambda(F op)
{
  // Create the callable on the heap by moving from op. Use a shared pointer to manage its lifetime.
  auto sp = std::make_shared<F>(std::move(op));

  // Use the copyable closure as the proxy of the move-only callable.
  return
    [sp](auto&&... args) -> decltype(auto) { return (*sp)(std::forward<decltype(args)>(args)...); };
}

/**
 * @brief Determine the NVTX color and call index. They are used to identify tasks from different
 * pread/pwrite calls. Tasks from the same pread/pwrite call are given the same color and call
 * index. The call index is atomically incremented on each pread/pwrite call, and will wrap around
 * once it reaches the maximum value the integer type `std::uint64_t` can hold (this overflow
 * behavior is well-defined in C++). The color is picked from an internal color palette according to
 * the call index value.
 *
 * @return A pair of NVTX color and call index.
 */
inline const std::pair<const nvtx_color_type&, std::uint64_t> get_next_color_and_call_idx() noexcept
{
  static std::atomic_uint64_t call_counter{1ull};
  auto call_idx    = call_counter.fetch_add(1ull, std::memory_order_relaxed);
  auto& nvtx_color = NvtxManager::get_color_by_index(call_idx);
  return {nvtx_color, call_idx};
}

/**
 * @brief Options for a single I/O task submission.
 */
struct TaskOptions {
  /// Thread pool for task execution. Defaults to the global default thread pool.
  ThreadPool* thread_pool = &defaults::thread_pool();
  /// NVTX payload value for profiling annotations.
  std::uint64_t nvtx_payload = 0ull;
  /// NVTX color for profiling annotations.
  nvtx_color_type nvtx_color = NvtxManager::default_color();
};

/**
 * @brief Submit the task callable to the underlying thread pool.
 *
 * Both the callable and arguments shall satisfy copy-constructible.
 */
template <typename F, typename T>
std::future<std::size_t> submit_task(F op,
                                     T buf,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t devPtr_offset,
                                     TaskOptions opts = {})
{
  static_assert(std::is_invocable_r_v<std::size_t,
                                      decltype(op),
                                      decltype(buf),
                                      decltype(size),
                                      decltype(file_offset),
                                      decltype(devPtr_offset)>);
  expect_not_in_monitor();
  return opts.thread_pool->submit_task([=] {
    KVIKIO_NVTX_SCOPED_RANGE("task", opts.nvtx_payload, opts.nvtx_color);
    return op(buf, size, file_offset, devPtr_offset);
  });
}

/**
 * @brief Submit the move-only task callable to the underlying thread pool.
 *
 * @tparam F Callable type. F shall be move-only and have no argument.
 * @param op Callable.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename F>
std::future<std::size_t> submit_move_only_task(F op_move_only, TaskOptions opts = {})
{
  static_assert(std::is_invocable_r_v<std::size_t, F>);
  expect_not_in_monitor();
  auto op_copyable = make_copyable_lambda(std::move(op_move_only));
  return opts.thread_pool->submit_task([=] {
    KVIKIO_NVTX_SCOPED_RANGE("task", opts.nvtx_payload, opts.nvtx_color);
    return op_copyable();
  });
}

/**
 * @brief Options for parallel I/O execution.
 *
 * @see TaskOptions for per-task options derived via to_task_options().
 */
struct ParallelIoOptions {
  /// Thread pool for task execution. Defaults to the global default thread pool.
  ThreadPool* thread_pool = &defaults::thread_pool();
  /// NVTX call index for correlating tasks from the same pread/pwrite call.
  std::uint64_t call_idx = 0ull;
  /// NVTX color for profiling annotations.
  nvtx_color_type nvtx_color = NvtxManager::default_color();
  /// Size of the first task in bytes. If set, the first task uses this size instead of `task_size`,
  /// allowing callers to align subsequent tasks to page boundaries.
  std::optional<std::size_t> first_task_size = std::nullopt;
  /// Records the logical operation these tasks make up. Null when nobody is observing. The task
  /// that completes the work finishes it, so the observation lands before the caller's future.
  std::shared_ptr<LogicalObservationRecorder> recorder = nullptr;

  TaskOptions to_task_options() const noexcept
  {
    return {.thread_pool = thread_pool, .nvtx_payload = call_idx, .nvtx_color = nvtx_color};
  }
};

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
/**
 * @brief Failure points used to verify exceptional `parallel_io` task lifetime.
 */
enum class ParallelIoFailurePoint {
  TASK_SUBMISSION,
  FINAL_TASK_CONSTRUCTION,
  FINAL_TASK_SUBMISSION
};

namespace parallel_io_failure_injection {
inline constexpr auto disabled = std::numeric_limits<std::size_t>::max();
inline std::atomic<ParallelIoFailurePoint> point{ParallelIoFailurePoint::TASK_SUBMISSION};
inline std::atomic<std::size_t> countdown{disabled};
}  // namespace parallel_io_failure_injection

/**
 * @brief Inject a `std::bad_alloc` at a selected `parallel_io` failure point.
 *
 * At `TASK_SUBMISSION`, @p successful_reaches is the number of task submissions allowed before the
 * injected failure. The other points are reached once per call and normally use zero.
 */
inline void inject_parallel_io_failure_for_testing(ParallelIoFailurePoint point,
                                                   std::size_t successful_reaches = 0) noexcept
{
  parallel_io_failure_injection::point.store(point, std::memory_order_relaxed);
  parallel_io_failure_injection::countdown.store(successful_reaches, std::memory_order_release);
}

inline void maybe_inject_parallel_io_failure(ParallelIoFailurePoint point)
{
  using namespace parallel_io_failure_injection;
  if (parallel_io_failure_injection::point.load(std::memory_order_relaxed) != point) { return; }

  auto remaining = countdown.load(std::memory_order_acquire);
  while (remaining != disabled) {
    if (remaining == 0) {
      if (countdown.compare_exchange_weak(
            remaining, disabled, std::memory_order_acq_rel, std::memory_order_acquire)) {
        throw std::bad_alloc{};
      }
    } else if (countdown.compare_exchange_weak(
                 remaining, remaining - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }
}
#endif

/**
 * @brief Shared owner that drains submitted tasks before their target storage can unwind.
 *
 * `parallel_io` creates this object and reserves its complete capacity before submitting the first
 * task. The final waiter captures it by shared ownership. If a later submission or construction of
 * that waiter throws, stack unwinding releases the last owner and this destructor waits for every
 * task that was already issued.
 */
class ParallelIoSubmittedTasks {
 public:
  struct DrainResult {
    std::size_t bytes{};
    std::exception_ptr first_error{};
  };

  explicit ParallelIoSubmittedTasks(std::size_t capacity) { _tasks.reserve(capacity); }

  ParallelIoSubmittedTasks(ParallelIoSubmittedTasks const&)            = delete;
  ParallelIoSubmittedTasks& operator=(ParallelIoSubmittedTasks const&) = delete;

  ~ParallelIoSubmittedTasks() noexcept { std::ignore = drain(); }

  void push(std::future<std::size_t> task)
  {
    if (_tasks.size() == _tasks.capacity()) {
      // This is a defensive invariant check. Drain the otherwise-untracked task before reporting a
      // KvikIO logic error, then let this object's destructor drain all earlier tasks.
      try {
        std::ignore = task.get();
      } catch (...) {
      }
      KVIKIO_FAIL("parallel_io submitted-task capacity exhausted", std::logic_error);
    }
    try {
      _tasks.push_back(std::move(task));
    } catch (...) {
      auto const insertion_error = std::current_exception();
      if (task.valid()) {
        try {
          std::ignore = task.get();
        } catch (...) {
        }
      }
      std::rethrow_exception(insertion_error);
    }
  }

  [[nodiscard]] DrainResult drain() noexcept
  {
    DrainResult result;
    for (auto& task : _tasks) {
      if (!task.valid()) { continue; }
      try {
        result.bytes += task.get();
      } catch (...) {
        if (result.first_error == nullptr) { result.first_error = std::current_exception(); }
      }
    }
    _tasks.clear();
    return result;
  }

 private:
  std::vector<std::future<std::size_t>> _tasks;
};

/**
 * @brief Apply read or write operation in parallel.
 *
 * @tparam F The type of the function applying the read or write operation.
 * @tparam T The type of the memory pointer.
 * @param op The function applying the read or write operation.
 * @param buf Buffer pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param task_size Size of each task in bytes.
 * @param devPtr_offset Offset relative to the `devPtr_base` pointer, used only with registered
 * buffers.
 * @param opts Optional parameters for parallel execution. @see ParallelIoOptions.
 * @return A future to be used later to check if the operation has finished its execution.
 */
template <typename F, typename T>
std::future<std::size_t> parallel_io(F op,
                                     T buf,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t task_size,
                                     std::size_t devPtr_offset,
                                     ParallelIoOptions opts = {})
{
  KVIKIO_EXPECT(task_size > 0, "`task_size` must be positive", std::invalid_argument);
  KVIKIO_EXPECT(opts.thread_pool != nullptr, "The thread pool must not be nullptr");
  static_assert(std::is_invocable_r_v<std::size_t,
                                      decltype(op),
                                      decltype(buf),
                                      decltype(size),
                                      decltype(file_offset),
                                      decltype(devPtr_offset)>);

  // Single-task guard
  if (task_size >= size || get_page_size() >= size) {
    if (!opts.recorder) {
      return detail::submit_task(op, buf, size, file_offset, devPtr_offset, opts.to_task_options());
    }
    auto single_task = [op, rec = opts.recorder](
                         T b, std::size_t s, std::size_t fo, std::size_t dpo) -> std::size_t {
      try {
        auto const nbytes = op(b, s, fo, dpo);
        rec->finish(nbytes);
        return nbytes;
      } catch (...) {
        rec->finish_with_failure();
        throw;
      }
    };
    return detail::submit_task(
      single_task, buf, size, file_offset, devPtr_offset, opts.to_task_options());
  }

  auto const actual_first_task_size = opts.first_task_size.value_or(task_size);
  KVIKIO_EXPECT(actual_first_task_size > 0,
                "`first_task_size` must be positive when specified",
                std::invalid_argument);
  KVIKIO_EXPECT(actual_first_task_size <= task_size,
                "`first_task_size` must not exceed `task_size`",
                std::invalid_argument);
  auto const first_size            = std::min(actual_first_task_size, size);
  auto const remaining_after_first = size - first_size;
  auto const remaining_task_count =
    remaining_after_first == 0 ? std::size_t{0} : 1 + (remaining_after_first - 1) / task_size;
  auto const total_task_count = 1 + remaining_task_count;

  // Allocate the shared state and all future slots before the first task can touch caller-owned
  // storage. Every task except the final waiter is retained here.
  auto submitted_tasks =
    std::make_shared<ParallelIoSubmittedTasks>(total_task_count - std::size_t{1});

  // 1) Submit the first task (possibly shorter to satisfy caller alignment needs).
  auto cur_size = first_size;
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  maybe_inject_parallel_io_failure(ParallelIoFailurePoint::TASK_SUBMISSION);
#endif
  submitted_tasks->push(
    detail::submit_task(op, buf, cur_size, file_offset, devPtr_offset, opts.to_task_options()));
  file_offset += cur_size;
  devPtr_offset += cur_size;
  size -= cur_size;

  // 2) Submit remaining tasks but the last. These are all `task_size` sized.
  while (size > task_size) {
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
    maybe_inject_parallel_io_failure(ParallelIoFailurePoint::TASK_SUBMISSION);
#endif
    submitted_tasks->push(
      detail::submit_task(op, buf, task_size, file_offset, devPtr_offset, opts.to_task_options()));
    file_offset += task_size;
    devPtr_offset += task_size;
    size -= task_size;
  }

  // 3) Submit the last task, which consists of performing the last I/O and waiting the previous
  // tasks.
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  maybe_inject_parallel_io_failure(ParallelIoFailurePoint::FINAL_TASK_CONSTRUCTION);
#endif
  auto last_task =
    [op, buf, size, file_offset, devPtr_offset, submitted_tasks, rec = opts.recorder]() mutable
    -> std::size_t {
    // This task both performs the final part and waits for the others, so it is where the logical
    // operation actually completes, and therefore where its observation is emitted.
    std::size_t ret{};
    std::exception_ptr first_error;
    try {
      ret = op(buf, size, file_offset, devPtr_offset);
    } catch (...) {
      // Preserve the final task's failure, matching the existing execution order, but still wait
      // for every earlier task before exposing it.
      first_error = std::current_exception();
    }

    auto const previous = submitted_tasks->drain();
    ret += previous.bytes;
    if (first_error == nullptr) { first_error = previous.first_error; }

    if (first_error != nullptr) {
      if (rec) { rec->finish_with_failure(); }
      std::rethrow_exception(first_error);
    }
    if (rec) { rec->finish(ret); }
    return ret;
  };
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  maybe_inject_parallel_io_failure(ParallelIoFailurePoint::FINAL_TASK_SUBMISSION);
#endif
  return detail::submit_move_only_task(std::move(last_task), opts.to_task_options());
}

}  // namespace kvikio::detail
