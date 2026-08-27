/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <kvikio/detail/direct_receive_cuda.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/error.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

#include "utils/utils.hpp"

namespace {

using kvikio::detail::DirectReceiveCudaAddResult;
using kvikio::detail::DirectReceiveCudaBatch;
using kvikio::detail::DirectReceiveCudaPath;
using kvikio::detail::DirectReceiveReleasedSlot;
using kvikio::detail::DirectReceiveSlotPool;
using kvikio::detail::DirectReceiveSpan;
using kvikio::detail::IoEventBarrier;

struct CudaDeviceDeleter {
  void operator()(void* pointer) const noexcept
  {
    if (pointer != nullptr) { std::ignore = cudaFree(pointer); }
  }
};

struct CudaStreamDeleter {
  void operator()(CUstream_st* stream) const noexcept
  {
    if (stream != nullptr) { std::ignore = cudaStreamDestroy(stream); }
  }
};

struct MemcpyCall {
  CUdeviceptr destination{};
  CUdeviceptr source{};
  std::size_t size{};
  CUstream stream{};

  friend bool operator==(MemcpyCall const& lhs, MemcpyCall const& rhs) noexcept
  {
    return lhs.destination == rhs.destination && lhs.source == rhs.source && lhs.size == rhs.size &&
           lhs.stream == rhs.stream;
  }
};

struct MockState {
  std::vector<MemcpyCall> memcpy_calls;
  CUresult memcpy_result{CUDA_SUCCESS};
  std::size_t fail_memcpy_call{std::numeric_limits<std::size_t>::max()};
  CUresult memcpy_failure{CUDA_ERROR_OUT_OF_MEMORY};
  std::size_t stream_synchronize_calls{};
  CUresult stream_synchronize_result{CUDA_SUCCESS};
  bool forward_stream_synchronize{true};
  std::size_t event_record_calls{};
  std::size_t fail_event_record_call{std::numeric_limits<std::size_t>::max()};
  CUresult event_record_failure{CUDA_ERROR_INVALID_VALUE};
  std::size_t event_query_calls{};
  CUresult event_query_result{CUDA_SUCCESS};
  bool override_event_query{};
  std::size_t event_synchronize_calls{};
  CUresult event_synchronize_result{CUDA_SUCCESS};
  bool override_event_synchronize{};
  decltype(kvikio::cudaAPI::instance().StreamSynchronize) saved_stream_synchronize{};
  decltype(kvikio::cudaAPI::instance().EventRecord) saved_event_record{};
  decltype(kvikio::cudaAPI::instance().EventQuery) saved_event_query{};
  decltype(kvikio::cudaAPI::instance().EventSynchronize) saved_event_synchronize{};
};

MockState* active_mock{};

CUresult CUDAAPI record_memcpy(CUdeviceptr destination,
                               CUdeviceptr source,
                               std::size_t size,
                               CUstream stream)
{
  auto const call = active_mock->memcpy_calls.size();
  active_mock->memcpy_calls.push_back({destination, source, size, stream});
  return call == active_mock->fail_memcpy_call ? active_mock->memcpy_failure
                                               : active_mock->memcpy_result;
}

CUresult CUDAAPI record_stream_synchronize(CUstream stream)
{
  ++active_mock->stream_synchronize_calls;
  if (active_mock->stream_synchronize_result != CUDA_SUCCESS) {
    return active_mock->stream_synchronize_result;
  }
  return active_mock->forward_stream_synchronize ? active_mock->saved_stream_synchronize(stream)
                                                 : CUDA_SUCCESS;
}

CUresult CUDAAPI record_event(CUevent event, CUstream stream)
{
  auto const call = active_mock->event_record_calls++;
  if (call == active_mock->fail_event_record_call) { return active_mock->event_record_failure; }
  return active_mock->saved_event_record(event, stream);
}

CUresult CUDAAPI query_event(CUevent event)
{
  ++active_mock->event_query_calls;
  return active_mock->override_event_query ? active_mock->event_query_result
                                           : active_mock->saved_event_query(event);
}

CUresult CUDAAPI synchronize_event(CUevent event)
{
  ++active_mock->event_synchronize_calls;
  return active_mock->override_event_synchronize ? active_mock->event_synchronize_result
                                                 : active_mock->saved_event_synchronize(event);
}

class ScopedCudaOverrides {
 public:
  ScopedCudaOverrides()
    : _cuda{kvikio::cudaAPI::instance()},
      _saved_memcpy{_cuda.MemcpyAsync},
      _saved_batch{_cuda.MemcpyBatchAsync}
  {
    active_mock                     = &_state;
    _state.saved_stream_synchronize = _cuda.StreamSynchronize;
    _state.saved_event_record       = _cuda.EventRecord;
    _state.saved_event_query        = _cuda.EventQuery;
    _state.saved_event_synchronize  = _cuda.EventSynchronize;
    _cuda.MemcpyAsync               = &record_memcpy;
    _cuda.StreamSynchronize         = &record_stream_synchronize;
    _cuda.EventRecord               = &record_event;
    _cuda.EventQuery                = &query_event;
    _cuda.EventSynchronize          = &synchronize_event;
    _cuda.MemcpyBatchAsync.reset();
  }

  ScopedCudaOverrides(ScopedCudaOverrides const&)            = delete;
  ScopedCudaOverrides& operator=(ScopedCudaOverrides const&) = delete;

  ~ScopedCudaOverrides()
  {
    _cuda.MemcpyAsync       = _saved_memcpy;
    _cuda.StreamSynchronize = _state.saved_stream_synchronize;
    _cuda.EventRecord       = _state.saved_event_record;
    _cuda.EventQuery        = _state.saved_event_query;
    _cuda.EventSynchronize  = _state.saved_event_synchronize;
    _cuda.MemcpyBatchAsync  = std::move(_saved_batch);
    active_mock             = nullptr;
  }

  [[nodiscard]] MockState& state() noexcept { return _state; }

  void set_batch(kvikio::detail::AnyCallable batch) { _cuda.MemcpyBatchAsync = std::move(batch); }

 private:
  kvikio::cudaAPI& _cuda;
  decltype(kvikio::cudaAPI::instance().MemcpyAsync) _saved_memcpy;
  kvikio::detail::AnyCallable _saved_batch;
  MockState _state;
};

#if CUDA_VERSION >= 12080

struct BatchObservation {
  std::size_t calls{};
  std::size_t count{};
  std::array<CUdeviceptr, kvikio::detail::direct_receive_max_copies_per_cuda_batch> destinations{};
  std::array<CUdeviceptr, kvikio::detail::direct_receive_max_copies_per_cuda_batch> sources{};
  std::array<std::size_t, kvikio::detail::direct_receive_max_copies_per_cuda_batch> sizes{};
  CUmemcpyAttributes attribute{};
  std::size_t attribute_index{};
  std::size_t attribute_count{};
  CUstream stream{};
#if CUDA_VERSION < 13000
  std::size_t* failure_index{};
#endif
  CUresult result{CUDA_SUCCESS};
};

kvikio::detail::AnyCallable make_batch_recorder(BatchObservation& observation)
{
  kvikio::detail::AnyCallable batch;
  batch.set([&observation](CUdeviceptr* destinations,
                           CUdeviceptr* sources,
                           std::size_t* sizes,
                           std::size_t count,
                           CUmemcpyAttributes* attributes,
                           std::size_t* attribute_indexes,
                           std::size_t num_attributes,
#if CUDA_VERSION < 13000
                           std::size_t* failure_index,
#endif
                           CUstream stream) -> CUresult {
    ++observation.calls;
    observation.count = count;
    std::copy_n(destinations, count, observation.destinations.begin());
    std::copy_n(sources, count, observation.sources.begin());
    std::copy_n(sizes, count, observation.sizes.begin());
    observation.attribute_count = num_attributes;
    if (num_attributes != 0) {
      observation.attribute       = attributes[0];
      observation.attribute_index = attribute_indexes[0];
    }
    observation.stream = stream;
#if CUDA_VERSION < 13000
    observation.failure_index = failure_index;
#endif
    return observation.result;
  });
  return batch;
}

#endif

CUcontext current_context()
{
  CUcontext context{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetCurrent(&context));
  return context;
}

DirectReceiveReleasedSlot released_slot(std::initializer_list<DirectReceiveSpan> spans,
                                        std::size_t raw_bytes)
{
  DirectReceiveReleasedSlot released;
  released.raw_bytes  = raw_bytes;
  released.span_count = spans.size();
  std::size_t index{};
  for (auto const span : spans) {
    released.spans[index++] = span;
    released.body_bytes += span.size;
  }
  return released;
}

DirectReceiveReleasedSlot header_only_slot(std::size_t raw_bytes)
{
  DirectReceiveReleasedSlot released;
  released.raw_bytes = raw_bytes;
  return released;
}

class DirectReceiveCudaTest : public testing::Test {
 protected:
  void SetUp() override
  {
    KVIKIO_CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream{};
    KVIKIO_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    _stream.reset(stream);
  }

  [[nodiscard]] std::unique_ptr<DirectReceiveSlotPool> make_pool(std::size_t slots) const
  {
    auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
    return DirectReceiveSlotPool::create_for_testing(slot_size, slots * slot_size);
  }

  [[nodiscard]] std::shared_ptr<IoEventBarrier> make_barrier() const
  {
    return std::make_shared<IoEventBarrier>(current_context());
  }

  [[nodiscard]] CUstream stream() const noexcept { return _stream.get(); }

 private:
  std::unique_ptr<CUstream_st, CudaStreamDeleter> _stream;
};

TEST_F(DirectReceiveCudaTest, copies_multiple_slots_and_spans_and_retains_sources_until_completion)
{
  auto pool   = make_pool(2);
  auto first  = pool->try_acquire();
  auto second = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto* first_bytes  = static_cast<std::uint8_t*>(first->get());
  auto* second_bytes = static_cast<std::uint8_t*>(second->get());
  std::fill_n(first_bytes, first->size(), std::uint8_t{0});
  std::fill_n(second_bytes, second->size(), std::uint8_t{0});
  std::copy_n(std::array<std::uint8_t, 4>{1, 2, 3, 4}.begin(), 4, first_bytes + 3);
  std::copy_n(std::array<std::uint8_t, 2>{5, 6}.begin(), 2, second_bytes + 7);
  std::copy_n(std::array<std::uint8_t, 2>{7, 8}.begin(), 2, second_bytes + 19);

  void* destination_raw{};
  KVIKIO_CHECK_CUDA(cudaMalloc(&destination_raw, 8));
  std::unique_ptr<void, CudaDeviceDeleter> destination{destination_raw};
  KVIKIO_CHECK_CUDA(cudaMemset(destination.get(), 0, 8));

  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto first_released  = released_slot({{3, 0, 4}}, 7);
  auto second_released = released_slot({{7, 4, 2}, {19, 6, 2}}, 21);
  EXPECT_EQ(
    batch.try_add(
      *first, first_released, kvikio::convert_void2deviceptr(destination.get()), 8, barrier, 101),
    DirectReceiveCudaAddResult::added);
  EXPECT_EQ(
    batch.try_add(
      *second, second_released, kvikio::convert_void2deviceptr(destination.get()), 8, barrier, 202),
    DirectReceiveCudaAddResult::added);
  EXPECT_FALSE(static_cast<bool>(*first));
  EXPECT_FALSE(static_cast<bool>(*second));

  auto const submission = batch.submit();
  EXPECT_EQ(submission.slots, 2);
  EXPECT_EQ(submission.copies, 3);
  EXPECT_EQ(submission.bytes, 8);
  EXPECT_TRUE(submission.asynchronous);
  EXPECT_EQ(pool->snapshot().checked_out_slots, 2);

  batch.wait();
  EXPECT_EQ(pool->snapshot().checked_out_slots, 0);
  EXPECT_EQ(pool->snapshot().free_slots, 2);
  auto const cookies = batch.finished_cookies();
  ASSERT_EQ(cookies.size(), 2);
  EXPECT_EQ(cookies[0], 101);
  EXPECT_EQ(cookies[1], 202);
  EXPECT_NO_THROW(barrier->sync_all_events());

  std::array<std::uint8_t, 8> actual{};
  KVIKIO_CHECK_CUDA(
    cudaMemcpy(actual.data(), destination.get(), actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, (std::array<std::uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST_F(DirectReceiveCudaTest, flattens_one_bounded_batch_and_records_each_barrier_once)
{
  auto pool   = make_pool(2);
  auto first  = pool->try_acquire();
  auto second = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto const first_source  = kvikio::convert_void2deviceptr(first->get());
  auto const second_source = kvikio::convert_void2deviceptr(second->get());
  auto first_barrier       = make_barrier();
  auto second_barrier      = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto first_released  = released_slot({{1, 0, 2}}, 3);
  auto second_released = released_slot({{4, 2, 3}, {10, 5, 1}}, 11);

  EXPECT_EQ(batch.try_add(*first, first_released, 0x10000, 6, first_barrier, 1),
            DirectReceiveCudaAddResult::added);
  EXPECT_EQ(batch.try_add(*second, second_released, 0x10000, 6, second_barrier, 2),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  EXPECT_NO_THROW(std::ignore = batch.submit());
  ASSERT_EQ(overrides.state().memcpy_calls.size(), 3);
  EXPECT_EQ(overrides.state().memcpy_calls[0].destination, 0x10000);
  EXPECT_EQ(overrides.state().memcpy_calls[0].source, first_source + 1);
  EXPECT_EQ(overrides.state().memcpy_calls[0].size, 2);
  EXPECT_EQ(overrides.state().memcpy_calls[1].destination, 0x10002);
  EXPECT_EQ(overrides.state().memcpy_calls[1].source, second_source + 4);
  EXPECT_EQ(overrides.state().memcpy_calls[1].size, 3);
  EXPECT_EQ(overrides.state().memcpy_calls[2].destination, 0x10005);
  EXPECT_EQ(overrides.state().memcpy_calls[2].source, second_source + 10);
  EXPECT_EQ(overrides.state().memcpy_calls[2].size, 1);
  // One record for each aggregate barrier plus one private batch-completion record.
  EXPECT_EQ(overrides.state().event_record_calls, 3);
  batch.wait();
  EXPECT_NO_THROW(first_barrier->sync_all_events());
  EXPECT_NO_THROW(second_barrier->sync_all_events());
}

TEST_F(DirectReceiveCudaTest, rejects_overlap_and_malformed_descriptors_before_taking_ownership)
{
  auto pool      = make_pool(3);
  auto first     = pool->try_acquire();
  auto overlap   = pool->try_acquire();
  auto malformed = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(overlap.has_value());
  ASSERT_TRUE(malformed.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};

  auto first_released = released_slot({{0, 0, 4}}, 4);
  EXPECT_EQ(batch.try_add(*first, first_released, 0x20000, 8, barrier, 1),
            DirectReceiveCudaAddResult::added);

  auto overlapping = released_slot({{0, 0, 4}}, 4);
  EXPECT_THROW(std::ignore = batch.try_add(*overlap, overlapping, 0x20002, 4, barrier, 2),
               std::invalid_argument);
  EXPECT_TRUE(static_cast<bool>(*overlap));
  EXPECT_EQ(batch.slot_count(), 1);
  EXPECT_EQ(batch.copy_count(), 1);

  auto wrong_sum = released_slot({{0, 4, 2}}, 2);
  ++wrong_sum.body_bytes;
  EXPECT_THROW(std::ignore = batch.try_add(*malformed, wrong_sum, 0x20000, 8, barrier, 3),
               std::invalid_argument);
  EXPECT_TRUE(static_cast<bool>(*malformed));
  EXPECT_EQ(batch.slot_count(), 1);
}

TEST_F(DirectReceiveCudaTest, rejects_bounds_overflow_context_and_noncontiguous_output)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};

  auto source_oob = released_slot({{slot->size() - 1, 0, 2}}, slot->size());
  EXPECT_THROW(std::ignore = batch.try_add(*slot, source_oob, 0x1000, 2, barrier, 0),
               std::invalid_argument);
  EXPECT_TRUE(static_cast<bool>(*slot));

  auto destination_oob = released_slot({{0, 1, 2}}, 2);
  EXPECT_THROW(std::ignore = batch.try_add(*slot, destination_oob, 0x1000, 2, barrier, 0),
               std::invalid_argument);

  auto destination_overflow = released_slot({{0, 1, 1}}, 1);
  EXPECT_THROW(
    std::ignore = batch.try_add(
      *slot, destination_overflow, std::numeric_limits<CUdeviceptr>::max(), 2, barrier, 0),
    std::invalid_argument);

  auto noncontiguous = released_slot({{0, 0, 1}, {2, 2, 1}}, 3);
  EXPECT_THROW(std::ignore = batch.try_add(*slot, noncontiguous, 0x1000, 3, barrier, 0),
               std::invalid_argument);

  auto foreign_barrier = std::make_shared<IoEventBarrier>(reinterpret_cast<CUcontext>(0x1));
  auto valid           = released_slot({{0, 0, 1}}, 1);
  EXPECT_THROW(std::ignore = batch.try_add(*slot, valid, 0x1000, 1, foreign_barrier, 0),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = batch.try_add(*slot, valid, 0, 1, barrier, 0), std::invalid_argument);
  EXPECT_TRUE(static_cast<bool>(*slot));
  EXPECT_TRUE(batch.empty());
}

TEST_F(DirectReceiveCudaTest, rejects_an_invalid_accounting_path)
{
  EXPECT_THROW(
    DirectReceiveCudaBatch(current_context(), stream(), static_cast<DirectReceiveCudaPath>(255)),
    std::invalid_argument);
}

TEST_F(DirectReceiveCudaTest, fixed_capacity_never_splits_a_released_slot)
{
  constexpr auto capacity = kvikio::detail::direct_receive_max_copies_per_cuda_batch;
  auto pool               = make_pool(capacity + 1);
  auto barrier            = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  std::vector<std::optional<DirectReceiveSlotPool::Slot>> slots;
  slots.reserve(capacity + 1);
  for (std::size_t i = 0; i < capacity + 1; ++i) {
    slots.push_back(pool->try_acquire());
    ASSERT_TRUE(slots.back().has_value());
  }

  auto one = released_slot({{0, 0, 1}}, 1);
  for (std::size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(batch.try_add(*slots[i], one, 0x100000 + 2 * i, 1, barrier, i),
              DirectReceiveCudaAddResult::added);
  }
  EXPECT_EQ(batch.try_add(*slots.back(), one, 0x200000, 1, barrier, capacity),
            DirectReceiveCudaAddResult::batch_full);
  EXPECT_TRUE(static_cast<bool>(*slots.back()));
  EXPECT_EQ(batch.copy_count(), capacity);
  batch.reset();
  EXPECT_EQ(pool->snapshot().checked_out_slots, 1);
}

TEST_F(DirectReceiveCudaTest, one_slot_may_use_the_entire_copy_capacity)
{
  constexpr auto capacity = kvikio::detail::direct_receive_max_copies_per_cuda_batch;
  auto pool               = make_pool(1);
  auto slot               = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  DirectReceiveReleasedSlot released;
  released.span_count = capacity;
  released.raw_bytes  = 2 * capacity;
  released.body_bytes = capacity;
  for (std::size_t i = 0; i < capacity; ++i) {
    released.spans[i] = {.source_offset = 2 * i, .destination_offset = i, .size = 1};
  }
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*slot, released, 0x300000, capacity, make_barrier(), 7),
            DirectReceiveCudaAddResult::added);
  EXPECT_EQ(batch.slot_count(), 1);
  EXPECT_EQ(batch.copy_count(), capacity);
}

TEST_F(DirectReceiveCudaTest, header_only_batch_completes_without_cuda_submission)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*slot, header_only_slot(17), 0, 0, make_barrier(), 88),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  auto const submission = batch.submit();
  EXPECT_FALSE(submission.asynchronous);
  EXPECT_EQ(submission.copies, 0);
  EXPECT_TRUE(batch.completed());
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
  EXPECT_EQ(overrides.state().event_record_calls, 0);
  EXPECT_EQ(pool->snapshot().free_slots, 1);
  ASSERT_EQ(batch.finished_cookies().size(), 1);
  EXPECT_EQ(batch.finished_cookies()[0], 88);
}

TEST_F(DirectReceiveCudaTest, accounts_each_successful_logical_batch_to_its_fixed_path)
{
  auto pool        = make_pool(2);
  auto strict_slot = pool->try_acquire();
  auto copied_slot = pool->try_acquire();
  ASSERT_TRUE(strict_slot.has_value());
  ASSERT_TRUE(copied_slot.has_value());
  auto released = released_slot({{0, 0, 3}}, 3);

  DirectReceiveCudaBatch strict{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  DirectReceiveCudaBatch copied{current_context(), stream(), DirectReceiveCudaPath::copied_stream};
  EXPECT_EQ(strict.try_add(*strict_slot, released, 0x310000, 3, make_barrier(), 1),
            DirectReceiveCudaAddResult::added);
  EXPECT_EQ(copied.try_add(*copied_slot, released, 0x320000, 3, make_barrier(), 2),
            DirectReceiveCudaAddResult::added);

  kvikio::reset_remote_direct_receive_stats();
  ScopedCudaOverrides overrides;
  EXPECT_NO_THROW(std::ignore = strict.submit());
  EXPECT_NO_THROW(std::ignore = copied.submit());
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 3);
  EXPECT_EQ(stats.strict_rx_h2d_batches, 1);
  EXPECT_EQ(stats.copied_stream_h2d_bytes, 3);
  EXPECT_EQ(stats.copied_stream_h2d_batches, 1);
  strict.wait();
  copied.wait();
}

TEST_F(DirectReceiveCudaTest, pre_submit_allocation_failure_queues_no_copy_and_is_retryable)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto released = released_slot({{0, 0, 1}}, 1);
  EXPECT_EQ(batch.try_add(*slot, released, 0x400000, 1, make_barrier(), 0),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  DirectReceiveCudaBatch::inject_pre_submit_allocation_failure_for_testing();
  EXPECT_THROW(std::ignore = batch.submit(), std::bad_alloc);
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
  EXPECT_FALSE(batch.submitted());
  EXPECT_EQ(pool->snapshot().checked_out_slots, 1);
  EXPECT_NO_THROW(std::ignore = batch.submit());
  batch.wait();
  EXPECT_EQ(pool->snapshot().free_slots, 1);
}

TEST_F(DirectReceiveCudaTest, copy_failure_synchronizes_before_recycling_and_preserves_error)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto released = released_slot({{0, 0, 1}}, 1);
  EXPECT_EQ(batch.try_add(*slot, released, 0x500000, 1, make_barrier(), 0),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  overrides.state().memcpy_result = CUDA_ERROR_OUT_OF_MEMORY;
  EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_TRUE(batch.failed());
  ASSERT_EQ(batch.finished_cookies().size(), 1);
  EXPECT_EQ(batch.finished_cookies()[0], 0);
  EXPECT_EQ(pool->snapshot().checked_out_slots, 0);
  EXPECT_EQ(pool->snapshot().free_slots, 1);
  EXPECT_EQ(pool->snapshot().quarantined_slots, 0);
}

TEST_F(DirectReceiveCudaTest, fallback_prefix_failure_fences_before_recycling_all_slots)
{
  auto pool   = make_pool(2);
  auto first  = pool->try_acquire();
  auto second = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto const first_source  = kvikio::convert_void2deviceptr(first->get());
  auto const second_source = kvikio::convert_void2deviceptr(second->get());
  auto barrier             = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*first, released_slot({{1, 0, 2}}, 3), 0x510000, 6, barrier, 10),
            DirectReceiveCudaAddResult::added);
  EXPECT_EQ(
    batch.try_add(*second, released_slot({{4, 2, 3}, {10, 5, 1}}, 11), 0x510000, 6, barrier, 20),
    DirectReceiveCudaAddResult::added);

  kvikio::reset_remote_direct_receive_stats();
  ScopedCudaOverrides overrides;
  overrides.state().memcpy_calls.reserve(3);
  overrides.state().fail_memcpy_call           = 1;
  overrides.state().forward_stream_synchronize = false;
  EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
  ASSERT_EQ(overrides.state().memcpy_calls.size(), 2);
  EXPECT_EQ(overrides.state().memcpy_calls[0],
            (MemcpyCall{0x510000, first_source + 1, 2, stream()}));
  EXPECT_EQ(overrides.state().memcpy_calls[1],
            (MemcpyCall{0x510002, second_source + 4, 3, stream()}));
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_EQ(overrides.state().event_record_calls, 0);
  EXPECT_TRUE(batch.failed());
  ASSERT_EQ(batch.finished_cookies().size(), 2);
  EXPECT_EQ(batch.finished_cookies()[0], 10);
  EXPECT_EQ(batch.finished_cookies()[1], 20);
  EXPECT_EQ(pool->snapshot().checked_out_slots, 0);
  EXPECT_EQ(pool->snapshot().free_slots, 2);
  EXPECT_EQ(pool->snapshot().quarantined_slots, 0);
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 0);
  EXPECT_EQ(stats.strict_rx_h2d_batches, 0);
}

#if CUDA_VERSION >= 12080

TEST_F(DirectReceiveCudaTest, native_batch_failure_fences_before_recycling_all_slots)
{
  BatchObservation observation;
  ScopedCudaOverrides overrides;
  observation.result = CUDA_ERROR_OUT_OF_MEMORY;
  overrides.set_batch(make_batch_recorder(observation));
  overrides.state().forward_stream_synchronize = false;

  auto pool   = make_pool(2);
  auto first  = pool->try_acquire();
  auto second = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto const first_source  = kvikio::convert_void2deviceptr(first->get());
  auto const second_source = kvikio::convert_void2deviceptr(second->get());
  auto barrier             = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*first, released_slot({{1, 0, 2}}, 3), 0x520000, 6, barrier, 10),
            DirectReceiveCudaAddResult::added);
  EXPECT_EQ(
    batch.try_add(*second, released_slot({{4, 2, 3}, {10, 5, 1}}, 11), 0x520000, 6, barrier, 20),
    DirectReceiveCudaAddResult::added);

  EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
  EXPECT_EQ(observation.calls, 1);
  ASSERT_EQ(observation.count, 3);
  EXPECT_EQ(observation.destinations[0], 0x520000);
  EXPECT_EQ(observation.destinations[1], 0x520002);
  EXPECT_EQ(observation.destinations[2], 0x520005);
  EXPECT_EQ(observation.sources[0], first_source + 1);
  EXPECT_EQ(observation.sources[1], second_source + 4);
  EXPECT_EQ(observation.sources[2], second_source + 10);
  EXPECT_EQ(observation.sizes[0], 2);
  EXPECT_EQ(observation.sizes[1], 3);
  EXPECT_EQ(observation.sizes[2], 1);
  EXPECT_EQ(observation.stream, stream());
  EXPECT_EQ(observation.attribute_count, 1);
  EXPECT_EQ(observation.attribute.srcAccessOrder, CU_MEMCPY_SRC_ACCESS_ORDER_STREAM);
  EXPECT_EQ(observation.attribute_index, 0);
#if CUDA_VERSION < 13000
  EXPECT_EQ(observation.failure_index, nullptr);
#endif
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_EQ(overrides.state().event_record_calls, 0);
  EXPECT_TRUE(batch.failed());
  ASSERT_EQ(batch.finished_cookies().size(), 2);
  EXPECT_EQ(batch.finished_cookies()[0], 10);
  EXPECT_EQ(batch.finished_cookies()[1], 20);
  EXPECT_EQ(pool->snapshot().free_slots, 2);
  EXPECT_EQ(pool->snapshot().quarantined_slots, 0);
}

#endif

TEST_F(DirectReceiveCudaTest, failed_recovery_fence_quarantines_sources_and_marks_barrier_unknown)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto released = released_slot({{0, 0, 1}}, 1);
  EXPECT_EQ(batch.try_add(*slot, released, 0x600000, 1, barrier, 0),
            DirectReceiveCudaAddResult::added);

  {
    ScopedCudaOverrides overrides;
    overrides.state().memcpy_result             = CUDA_ERROR_OUT_OF_MEMORY;
    overrides.state().stream_synchronize_result = CUDA_ERROR_INVALID_CONTEXT;
    EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
    EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  }
  EXPECT_TRUE(batch.failed());
  ASSERT_EQ(batch.finished_cookies().size(), 1);
  auto const snapshot = pool->snapshot();
  EXPECT_EQ(snapshot.checked_out_slots, 0);
  EXPECT_EQ(snapshot.free_slots, 0);
  EXPECT_EQ(snapshot.quarantined_slots, 1);

  // The failed stream fence makes the aggregate barrier establish destination safety later.
  EXPECT_NO_THROW(barrier->sync_all_events());
}

TEST_F(DirectReceiveCudaTest, event_record_failure_uses_the_same_stream_recovery)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto released = released_slot({{0, 0, 1}}, 1);
  EXPECT_EQ(batch.try_add(*slot, released, 0x700000, 1, barrier, 0),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  kvikio::reset_remote_direct_receive_stats();
  overrides.state().fail_event_record_call = 0;
  EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
  EXPECT_EQ(overrides.state().memcpy_calls.size(), 1);
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_EQ(pool->snapshot().free_slots, 1);
  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 1);
  EXPECT_EQ(stats.strict_rx_h2d_batches, 1);
}

TEST_F(DirectReceiveCudaTest, private_completion_event_failure_uses_stream_recovery)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*slot, released_slot({{0, 0, 1}}, 1), 0x710000, 1, barrier, 7),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  overrides.state().fail_event_record_call = 1;
  EXPECT_THROW(std::ignore = batch.submit(), kvikio::CUfileException);
  EXPECT_EQ(overrides.state().event_record_calls, 2);
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_TRUE(batch.failed());
  ASSERT_EQ(batch.finished_cookies().size(), 1);
  EXPECT_EQ(batch.finished_cookies()[0], 7);
  EXPECT_EQ(pool->snapshot().free_slots, 1);
  EXPECT_NO_THROW(barrier->sync_all_events());
}

TEST_F(DirectReceiveCudaTest, incomplete_poll_retains_slots_until_a_later_wait)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  auto released = released_slot({{0, 0, 1}}, 1);
  EXPECT_EQ(batch.try_add(*slot, released, 0x800000, 1, make_barrier(), 0),
            DirectReceiveCudaAddResult::added);

  {
    ScopedCudaOverrides overrides;
    EXPECT_NO_THROW(std::ignore = batch.submit());
    overrides.state().override_event_query = true;
    overrides.state().event_query_result   = CUDA_ERROR_NOT_READY;
    EXPECT_FALSE(batch.poll());
    EXPECT_EQ(overrides.state().event_query_calls, 1);
    EXPECT_EQ(pool->snapshot().checked_out_slots, 1);
  }
  batch.wait();
  EXPECT_EQ(pool->snapshot().free_slots, 1);
}

TEST_F(DirectReceiveCudaTest, event_query_failure_fences_before_recycling)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*slot, released_slot({{0, 0, 1}}, 1), 0x810000, 1, barrier, 8),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  EXPECT_NO_THROW(std::ignore = batch.submit());
  overrides.state().override_event_query = true;
  overrides.state().event_query_result   = CUDA_ERROR_INVALID_VALUE;
  EXPECT_THROW(std::ignore = batch.poll(), kvikio::CUfileException);
  EXPECT_EQ(overrides.state().event_query_calls, 1);
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_TRUE(batch.failed());
  EXPECT_EQ(pool->snapshot().free_slots, 1);
}

TEST_F(DirectReceiveCudaTest, event_wait_failure_fences_before_recycling)
{
  auto pool = make_pool(1);
  auto slot = pool->try_acquire();
  ASSERT_TRUE(slot.has_value());
  auto barrier = make_barrier();
  DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
  EXPECT_EQ(batch.try_add(*slot, released_slot({{0, 0, 1}}, 1), 0x820000, 1, barrier, 9),
            DirectReceiveCudaAddResult::added);

  ScopedCudaOverrides overrides;
  EXPECT_NO_THROW(std::ignore = batch.submit());
  overrides.state().override_event_synchronize = true;
  overrides.state().event_synchronize_result   = CUDA_ERROR_INVALID_VALUE;
  EXPECT_THROW(batch.wait(), kvikio::CUfileException);
  EXPECT_EQ(overrides.state().event_synchronize_calls, 1);
  EXPECT_EQ(overrides.state().stream_synchronize_calls, 1);
  EXPECT_TRUE(batch.failed());
  EXPECT_EQ(pool->snapshot().free_slots, 1);
}

TEST_F(DirectReceiveCudaTest, destruction_fences_an_inflight_batch_before_recycling)
{
  auto pool = make_pool(1);
  {
    auto slot = pool->try_acquire();
    ASSERT_TRUE(slot.has_value());
    DirectReceiveCudaBatch batch{current_context(), stream(), DirectReceiveCudaPath::strict_rx};
    auto released = released_slot({{0, 0, 1}}, 1);
    EXPECT_EQ(batch.try_add(*slot, released, 0x900000, 1, make_barrier(), 0),
              DirectReceiveCudaAddResult::added);
    ScopedCudaOverrides overrides;
    EXPECT_NO_THROW(std::ignore = batch.submit());
    EXPECT_EQ(pool->snapshot().checked_out_slots, 1);
  }
  EXPECT_EQ(pool->snapshot().checked_out_slots, 0);
  EXPECT_EQ(pool->snapshot().free_slots, 1);
}

}  // namespace
