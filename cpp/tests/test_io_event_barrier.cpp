/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <future>
#include <latch>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

#include "utils/utils.hpp"

namespace {

std::atomic<int> event_synchronize_calls{0};
std::atomic<int> fail_event_synchronize_call{-1};
decltype(kvikio::cudaAPI::instance().EventSynchronize) saved_event_synchronize{};
bool fail_next_event_record{false};
CUevent failed_event_record{};
CUevent successful_event_record{};
decltype(kvikio::cudaAPI::instance().EventRecord) saved_event_record{};
std::atomic<int> context_synchronize_calls{0};
decltype(kvikio::cudaAPI::instance().CtxSynchronize) saved_context_synchronize{};

CUresult counting_event_synchronize(CUevent event)
{
  auto const call = event_synchronize_calls.fetch_add(1);
  if (call == fail_event_synchronize_call.load()) { return CUDA_ERROR_INVALID_VALUE; }
  return saved_event_synchronize(event);
}

CUresult fail_first_then_forward_event_record(CUevent event, CUstream stream)
{
  if (fail_next_event_record) {
    fail_next_event_record = false;
    failed_event_record    = event;
    return CUDA_ERROR_INVALID_VALUE;
  }
  successful_event_record = event;
  return saved_event_record(event, stream);
}

CUresult counting_context_synchronize()
{
  ++context_synchronize_calls;
  return saved_context_synchronize();
}

CUresult failing_context_synchronize() { return CUDA_ERROR_INVALID_CONTEXT; }

class ScopedEventSynchronizeCounter {
 public:
  ScopedEventSynchronizeCounter()
  {
    auto& slot                  = kvikio::cudaAPI::instance().EventSynchronize;
    saved_event_synchronize     = slot;
    event_synchronize_calls     = 0;
    fail_event_synchronize_call = -1;
    slot                        = &counting_event_synchronize;
  }

  ScopedEventSynchronizeCounter(ScopedEventSynchronizeCounter const&)            = delete;
  ScopedEventSynchronizeCounter& operator=(ScopedEventSynchronizeCounter const&) = delete;

  ~ScopedEventSynchronizeCounter()
  {
    kvikio::cudaAPI::instance().EventSynchronize = saved_event_synchronize;
  }
};

class ScopedFirstEventRecordFailure {
 public:
  ScopedFirstEventRecordFailure()
  {
    auto& slot              = kvikio::cudaAPI::instance().EventRecord;
    saved_event_record      = slot;
    fail_next_event_record  = true;
    failed_event_record     = nullptr;
    successful_event_record = nullptr;
    slot                    = &fail_first_then_forward_event_record;
  }

  ScopedFirstEventRecordFailure(ScopedFirstEventRecordFailure const&)            = delete;
  ScopedFirstEventRecordFailure& operator=(ScopedFirstEventRecordFailure const&) = delete;

  ~ScopedFirstEventRecordFailure() { kvikio::cudaAPI::instance().EventRecord = saved_event_record; }
};

class ScopedContextSynchronizeCounter {
 public:
  ScopedContextSynchronizeCounter()
  {
    auto& slot                = kvikio::cudaAPI::instance().CtxSynchronize;
    saved_context_synchronize = slot;
    context_synchronize_calls = 0;
    slot                      = &counting_context_synchronize;
  }

  ScopedContextSynchronizeCounter(ScopedContextSynchronizeCounter const&)            = delete;
  ScopedContextSynchronizeCounter& operator=(ScopedContextSynchronizeCounter const&) = delete;

  ~ScopedContextSynchronizeCounter()
  {
    kvikio::cudaAPI::instance().CtxSynchronize = saved_context_synchronize;
  }
};

CUcontext current_context()
{
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetCurrent(&ctx));
  return ctx;
}

}  // namespace

class IoEventBarrierTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
};

TEST_F(IoEventBarrierTest, cuda_context_stored)
{
  auto ctx = current_context();
  kvikio::detail::IoEventBarrier barrier(ctx);
  EXPECT_EQ(barrier.cuda_context(), ctx);
}

TEST_F(IoEventBarrierTest, sync_with_no_records_is_noop)
{
  kvikio::detail::IoEventBarrier barrier(current_context());
  // No slots, nothing to wait for.
  EXPECT_NO_THROW(barrier.sync_all_events());
}

TEST_F(IoEventBarrierTest, single_thread_record_and_sync)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  EXPECT_NO_THROW({
    barrier.record_event(stream);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, split_prepare_and_record_uses_the_reserved_event)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  EXPECT_THROW(barrier.record_prepared_event(stream), std::logic_error);
  EXPECT_NO_THROW({
    barrier.prepare_event(stream);
    barrier.prepare_event(stream);
    barrier.record_prepared_event(stream);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, split_prepare_is_bound_to_the_exact_stream)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream first{};
  CUstream second{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&first, CU_STREAM_DEFAULT));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&second, CU_STREAM_DEFAULT));

  barrier.prepare_event(first);
  EXPECT_THROW(barrier.record_prepared_event(second), std::logic_error);
  EXPECT_NO_THROW({
    barrier.record_prepared_event(first);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(first));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(second));
}

TEST_F(IoEventBarrierTest, same_thread_records_distinct_events_for_distinct_streams)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream first{};
  CUstream second{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&first, CU_STREAM_DEFAULT));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&second, CU_STREAM_DEFAULT));
  barrier.record_event(first);
  barrier.record_event(second);

  {
    ScopedEventSynchronizeCounter const counter;
    EXPECT_NO_THROW(barrier.sync_all_events());
    EXPECT_EQ(event_synchronize_calls.load(), 2);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(first));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(second));
}

TEST_F(IoEventBarrierTest, failed_record_is_abandoned_and_forces_context_fence)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  {
    ScopedFirstEventRecordFailure const record_failure;
    ScopedContextSynchronizeCounter const context_counter;
    EXPECT_THROW(barrier.record_event(stream), kvikio::CUfileException);
    EXPECT_NO_THROW(barrier.record_event(stream));
    EXPECT_NE(failed_event_record, nullptr);
    EXPECT_NE(successful_event_record, nullptr);
    EXPECT_NE(failed_event_record, successful_event_record);
    EXPECT_NO_THROW(barrier.sync_all_events());
    EXPECT_EQ(context_synchronize_calls.load(), 1);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, failed_io_still_synchronizes_before_rethrow)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  barrier.record_event(stream);

  std::promise<std::size_t> promise;
  auto completion = promise.get_future();
  promise.set_exception(std::make_exception_ptr(std::runtime_error{"injected I/O failure"}));

  {
    ScopedEventSynchronizeCounter const counter;
    EXPECT_THROW(
      {
        try {
          std::ignore = kvikio::detail::wait_for_io_completion(std::move(completion), barrier);
        } catch (std::runtime_error const& error) {
          EXPECT_STREQ(error.what(), "injected I/O failure");
          throw;
        }
      },
      std::runtime_error);
    EXPECT_EQ(event_synchronize_calls.load(), 1);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, io_error_precedes_secondary_event_error_after_context_fence)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  barrier.record_event(stream);

  std::promise<std::size_t> promise;
  auto completion = promise.get_future();
  promise.set_exception(std::make_exception_ptr(std::runtime_error{"injected I/O failure"}));

  {
    ScopedEventSynchronizeCounter const event_counter;
    ScopedContextSynchronizeCounter const context_counter;
    fail_event_synchronize_call = 0;
    EXPECT_THROW(
      {
        try {
          std::ignore = kvikio::detail::wait_for_io_completion(std::move(completion), barrier);
        } catch (std::runtime_error const& error) {
          EXPECT_STREQ(error.what(), "injected I/O failure");
          throw;
        }
      },
      std::runtime_error);
    EXPECT_EQ(event_synchronize_calls.load(), 1);
    EXPECT_EQ(context_synchronize_calls.load(), 1);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, re_record_overwrites_same_slot)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  // Multiple records on the same thread and stream reuse the same slot. sync_all_events should
  // still succeed after the final re-record.
  EXPECT_NO_THROW({
    barrier.record_event(stream);
    barrier.record_event(stream);
    barrier.record_event(stream);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, sync_is_context_agnostic)
{
  auto ctx = current_context();
  ASSERT_NE(ctx, nullptr);

  CUdevice dev{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetDevice(&dev));

  kvikio::detail::IoEventBarrier barrier(ctx);

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  barrier.record_event(stream);

  // Case 1: no context current on the calling thread.
  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  ASSERT_EQ(popped, ctx);
  ASSERT_EQ(current_context(), nullptr);
  EXPECT_NO_THROW(barrier.sync_all_events());

  // Case 2: a different context current on the calling thread.
  CUcontext other_ctx{};
#if CUDA_VERSION >= 13000
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&other_ctx, nullptr, 0, dev));
#else
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&other_ctx, 0, dev));
#endif
  ASSERT_NE(other_ctx, ctx);
  ASSERT_EQ(current_context(), other_ctx);
  EXPECT_NO_THROW(barrier.sync_all_events());

  // Restore the primary context.
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  ASSERT_EQ(popped, other_ctx);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxDestroy(other_ctx));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPushCurrent(ctx));
  ASSERT_EQ(current_context(), ctx);

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, multi_thread_record_then_sync_on_caller)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  constexpr int num_workers = 4;
  std::atomic<int> errors{0};
  std::latch records_finished{num_workers};
  std::latch release_workers{1};
  std::vector<std::thread> workers;
  workers.reserve(num_workers);

  for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([&] {
      try {
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        barrier.record_event(stream);
      } catch (...) {
        ++errors;
      }
      records_finished.count_down();
      release_workers.wait();
    });
  }
  records_finished.wait();
  release_workers.count_down();
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_EQ(errors.load(), 0);

  EXPECT_NO_THROW(barrier.sync_all_events());

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, event_failure_attempts_every_event_then_fences_context)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  constexpr int num_workers = 4;
  std::atomic<int> errors{0};
  std::latch records_finished{num_workers};
  std::latch release_workers{1};
  std::vector<std::thread> workers;
  workers.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([&] {
      try {
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        barrier.record_event(stream);
      } catch (...) {
        ++errors;
      }
      records_finished.count_down();
      release_workers.wait();
    });
  }
  records_finished.wait();
  release_workers.count_down();
  for (auto& worker : workers) {
    worker.join();
  }
  ASSERT_EQ(errors.load(), 0);

  {
    ScopedEventSynchronizeCounter const event_counter;
    ScopedContextSynchronizeCounter const context_counter;
    fail_event_synchronize_call = 0;
    EXPECT_THROW(barrier.sync_all_events(), kvikio::CUfileException);
    EXPECT_EQ(event_synchronize_calls.load(), num_workers);
    EXPECT_EQ(context_synchronize_calls.load(), 1);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, unknown_completion_forces_context_fence)
{
  kvikio::detail::IoEventBarrier barrier(current_context());
  barrier.mark_completion_unknown();

  ScopedContextSynchronizeCounter const context_counter;
  EXPECT_NO_THROW(barrier.sync_all_events());
  EXPECT_EQ(context_synchronize_calls.load(), 1);
}

TEST_F(IoEventBarrierTest, failed_last_resort_context_fence_terminates)
{
  kvikio::detail::IoEventBarrier barrier(current_context());
  barrier.mark_completion_unknown();

  EXPECT_DEATH(
    {
      kvikio::cudaAPI::instance().CtxSynchronize = &failing_context_synchronize;
      barrier.sync_all_events();
    },
    "");
}
