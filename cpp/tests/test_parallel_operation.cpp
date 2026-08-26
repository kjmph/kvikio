/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <latch>
#include <stdexcept>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/threadpool_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace {

using FailurePoint = kvikio::detail::ParallelIoFailurePoint;

struct ParallelIoFailureCase {
  FailurePoint point;
  std::size_t successful_reaches;
  std::size_t submitted_tasks;
  char const* name;
};

class ParallelIoFailureTest : public testing::TestWithParam<ParallelIoFailureCase> {};

TEST_P(ParallelIoFailureTest, drains_every_submitted_task_before_preserving_submission_error)
{
  auto const test_case = GetParam();
  auto const page_size = kvikio::get_page_size();
  auto const read_size = page_size * 3;

  kvikio::ThreadPool thread_pool{2};
  std::atomic<std::size_t> started{};
  std::atomic<std::size_t> completed{};
  std::promise<void> all_started_promise;
  auto all_started = all_started_promise.get_future();
  std::latch release_tasks{1};

  auto operation = [&](void*, std::size_t size, std::size_t, std::size_t) -> std::size_t {
    if (started.fetch_add(1, std::memory_order_relaxed) + 1 == test_case.submitted_tasks) {
      all_started_promise.set_value();
    }
    release_tasks.wait();
    completed.fetch_add(1, std::memory_order_relaxed);
    return size;
  };

  kvikio::detail::inject_parallel_io_failure_for_testing(test_case.point,
                                                         test_case.successful_reaches);
  auto invocation = std::async(std::launch::async, [&]() -> std::string {
    try {
      auto completion = kvikio::detail::parallel_io(operation,
                                                    static_cast<void*>(nullptr),
                                                    read_size,
                                                    0,
                                                    page_size,
                                                    0,
                                                    {.thread_pool = &thread_pool});
      std::ignore     = completion.get();
      return "parallel_io unexpectedly succeeded";
    } catch (std::bad_alloc const&) {
      return "bad_alloc";
    } catch (std::exception const& error) {
      return error.what();
    }
  });

  auto constexpr timeout = std::chrono::seconds{10};
  if (all_started.wait_for(timeout) != std::future_status::ready) {
    release_tasks.count_down();
    auto const outcome = invocation.get();
    FAIL() << "submitted tasks did not start; outcome: " << outcome;
  }

  EXPECT_EQ(invocation.wait_for(std::chrono::seconds{0}), std::future_status::timeout);
  release_tasks.count_down();
  EXPECT_EQ(invocation.get(), "bad_alloc");
  EXPECT_EQ(started.load(std::memory_order_relaxed), test_case.submitted_tasks);
  EXPECT_EQ(completed.load(std::memory_order_relaxed), test_case.submitted_tasks);
}

INSTANTIATE_TEST_SUITE_P(
  SubmissionAndFinalWaiterFailures,
  ParallelIoFailureTest,
  testing::Values(
    ParallelIoFailureCase{FailurePoint::TASK_SUBMISSION, 1, 1, "later_submission"},
    ParallelIoFailureCase{FailurePoint::FINAL_TASK_CONSTRUCTION, 0, 2, "final_construction"},
    ParallelIoFailureCase{FailurePoint::FINAL_TASK_SUBMISSION, 0, 2, "final_submission"}),
  [](testing::TestParamInfo<ParallelIoFailureCase> const& info) { return info.param.name; });

TEST(ParallelIoTaskSizing, rejects_first_task_larger_than_regular_task_before_submission)
{
  auto const page_size = kvikio::get_page_size();
  std::atomic<std::size_t> calls{};
  kvikio::ThreadPool thread_pool{1};
  auto operation = [&](void*, std::size_t size, std::size_t, std::size_t) {
    ++calls;
    return size;
  };

  EXPECT_THROW(std::ignore = kvikio::detail::parallel_io(
                 operation,
                 static_cast<void*>(nullptr),
                 page_size * 3,
                 0,
                 page_size,
                 0,
                 {.thread_pool = &thread_pool, .first_task_size = page_size * 2}),
               std::invalid_argument);
  EXPECT_EQ(calls.load(), 0);
}

TEST(ParallelIoCompletion, prior_task_failure_waits_for_every_task_then_propagates)
{
  auto const page_size = kvikio::get_page_size();
  kvikio::ThreadPool thread_pool{3};
  std::promise<void> blocked_task_started_promise;
  auto blocked_task_started = blocked_task_started_promise.get_future();
  std::latch release_blocked_task{1};

  auto operation = [&](
                     void*, std::size_t size, std::size_t file_offset, std::size_t) -> std::size_t {
    if (file_offset == 0) { throw std::runtime_error{"first prior task failed"}; }
    if (file_offset == page_size) {
      blocked_task_started_promise.set_value();
      release_blocked_task.wait();
    }
    return size;
  };

  auto completion = kvikio::detail::parallel_io(operation,
                                                static_cast<void*>(nullptr),
                                                page_size * 3,
                                                0,
                                                page_size,
                                                0,
                                                {.thread_pool = &thread_pool});
  auto waiter     = std::async(std::launch::async, [completion = std::move(completion)]() mutable {
    try {
      std::ignore = completion.get();
      return std::string{"parallel_io unexpectedly succeeded"};
    } catch (std::exception const& error) {
      return std::string{error.what()};
    }
  });

  auto constexpr timeout = std::chrono::seconds{10};
  if (blocked_task_started.wait_for(timeout) != std::future_status::ready) {
    release_blocked_task.count_down();
    FAIL() << "the blocking prior task did not start; outcome: " << waiter.get();
  }
  EXPECT_EQ(waiter.wait_for(std::chrono::seconds{0}), std::future_status::timeout);
  release_blocked_task.count_down();
  EXPECT_EQ(waiter.get(), "first prior task failed");
}

TEST(ParallelIoCompletion, final_task_failure_waits_for_every_prior_task_then_propagates)
{
  auto const page_size = kvikio::get_page_size();
  kvikio::ThreadPool thread_pool{3};
  std::atomic<std::size_t> prior_tasks_started{};
  std::promise<void> all_prior_tasks_started_promise;
  auto all_prior_tasks_started = all_prior_tasks_started_promise.get_future();
  std::promise<void> final_task_started_promise;
  auto final_task_started = final_task_started_promise.get_future();
  std::latch release_prior_tasks{1};

  auto operation = [&](
                     void*, std::size_t size, std::size_t file_offset, std::size_t) -> std::size_t {
    if (file_offset < page_size * 2) {
      if (prior_tasks_started.fetch_add(1, std::memory_order_relaxed) + 1 == 2) {
        all_prior_tasks_started_promise.set_value();
      }
      release_prior_tasks.wait();
      return size;
    }
    final_task_started_promise.set_value();
    throw std::runtime_error{"final task failed"};
  };

  auto completion = kvikio::detail::parallel_io(operation,
                                                static_cast<void*>(nullptr),
                                                page_size * 3,
                                                0,
                                                page_size,
                                                0,
                                                {.thread_pool = &thread_pool});
  auto waiter     = std::async(std::launch::async, [completion = std::move(completion)]() mutable {
    try {
      std::ignore = completion.get();
      return std::string{"parallel_io unexpectedly succeeded"};
    } catch (std::exception const& error) {
      return std::string{error.what()};
    }
  });

  auto constexpr timeout = std::chrono::seconds{10};
  if (all_prior_tasks_started.wait_for(timeout) != std::future_status::ready ||
      final_task_started.wait_for(timeout) != std::future_status::ready) {
    release_prior_tasks.count_down();
    FAIL() << "parallel tasks did not all start; outcome: " << waiter.get();
  }
  EXPECT_EQ(waiter.wait_for(std::chrono::seconds{0}), std::future_status::timeout);
  release_prior_tasks.count_down();
  EXPECT_EQ(waiter.get(), "final task failed");
}

}  // namespace
