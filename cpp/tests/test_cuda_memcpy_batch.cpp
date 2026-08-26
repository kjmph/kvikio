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
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

#include "utils/utils.hpp"

namespace {

struct MemcpyCall {
  CUdeviceptr destination;
  CUdeviceptr source;
  std::size_t size;
  CUstream stream;
};

struct MockState {
  std::vector<MemcpyCall> memcpy_calls;
  std::size_t fail_at{std::numeric_limits<std::size_t>::max()};
  CUresult failure{CUDA_ERROR_INVALID_VALUE};
  std::vector<CUstream> synchronize_calls;
};

MockState* active_mock{};

CUresult CUDAAPI record_memcpy_async(CUdeviceptr destination,
                                     CUdeviceptr source,
                                     std::size_t size,
                                     CUstream stream)
{
  auto const call = active_mock->memcpy_calls.size();
  active_mock->memcpy_calls.push_back(MemcpyCall{destination, source, size, stream});
  return call == active_mock->fail_at ? active_mock->failure : CUDA_SUCCESS;
}

CUresult CUDAAPI record_stream_synchronize(CUstream stream)
{
  active_mock->synchronize_calls.push_back(stream);
  return CUDA_SUCCESS;
}

class ScopedCudaOverrides {
 public:
  ScopedCudaOverrides()
    : _cuda{kvikio::cudaAPI::instance()},
      _saved_memcpy{_cuda.MemcpyAsync},
      _saved_synchronize{_cuda.StreamSynchronize},
      _saved_batch{_cuda.MemcpyBatchAsync}
  {
    active_mock             = &_state;
    _cuda.MemcpyAsync       = &record_memcpy_async;
    _cuda.StreamSynchronize = &record_stream_synchronize;
  }

  ScopedCudaOverrides(ScopedCudaOverrides const&)            = delete;
  ScopedCudaOverrides& operator=(ScopedCudaOverrides const&) = delete;

  ~ScopedCudaOverrides()
  {
    _cuda.MemcpyAsync       = _saved_memcpy;
    _cuda.StreamSynchronize = _saved_synchronize;
    _cuda.MemcpyBatchAsync  = std::move(_saved_batch);
    active_mock             = nullptr;
  }

  void reset_batch() { _cuda.MemcpyBatchAsync.reset(); }

  void set_batch(kvikio::detail::AnyCallable batch) { _cuda.MemcpyBatchAsync = std::move(batch); }

  [[nodiscard]] MockState& state() { return _state; }

 private:
  kvikio::cudaAPI& _cuda;
  decltype(kvikio::cudaAPI::instance().MemcpyAsync) _saved_memcpy;
  decltype(kvikio::cudaAPI::instance().StreamSynchronize) _saved_synchronize;
  kvikio::detail::AnyCallable _saved_batch;
  MockState _state;
};

#if CUDA_VERSION >= 12080

struct BatchObservation {
  std::size_t calls{};
  std::vector<CUdeviceptr> destinations;
  std::vector<CUdeviceptr> sources;
  std::vector<std::size_t> sizes;
  std::vector<CUmemcpyAttributes> attributes;
  std::vector<std::size_t> attribute_indexes;
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
    observation.destinations.assign(destinations, destinations + count);
    observation.sources.assign(sources, sources + count);
    observation.sizes.assign(sizes, sizes + count);
    observation.attributes.assign(attributes, attributes + num_attributes);
    observation.attribute_indexes.assign(attribute_indexes, attribute_indexes + num_attributes);
    observation.stream = stream;
#if CUDA_VERSION < 13000
    observation.failure_index = failure_index;
#endif
    return observation.result;
  });
  return batch;
}

#endif

TEST(CudaMemcpyBatch, zero_count_is_a_noop_and_nonzero_count_requires_descriptors)
{
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(nullptr, nullptr, nullptr, 0, nullptr),
            CUDA_SUCCESS);

  ScopedCudaOverrides overrides;
  overrides.reset_batch();

  CUdeviceptr destinations[]{0x1000};
  CUdeviceptr sources[]{0x2000};
  std::size_t sizes[]{1};
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(nullptr, sources, sizes, 1, nullptr),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, nullptr, sizes, 1, nullptr),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, nullptr, 1, nullptr),
            CUDA_ERROR_INVALID_VALUE);
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
  EXPECT_TRUE(overrides.state().synchronize_calls.empty());
}

TEST(CudaMemcpyBatch, zero_sized_copies_are_ignored)
{
  ScopedCudaOverrides overrides;
#if CUDA_VERSION >= 12080
  BatchObservation observation;
  overrides.set_batch(make_batch_recorder(observation));
#else
  overrides.reset_batch();
#endif

  CUdeviceptr all_empty_destinations[]{0, 0, 0};
  CUdeviceptr all_empty_sources[]{0, 0, 0};
  std::size_t all_empty_sizes[]{0, 0, 0};
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(
              all_empty_destinations, all_empty_sources, all_empty_sizes, 3, CU_STREAM_PER_THREAD),
            CUDA_SUCCESS);
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());

  CUdeviceptr destinations[]{0x1000, 0, 0x3000};
  CUdeviceptr sources[]{0x4000, 0, 0x6000};
  std::size_t sizes[]{7, 0, 11};
  EXPECT_EQ(
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 3, CU_STREAM_PER_THREAD),
    CUDA_SUCCESS);

  ASSERT_EQ(overrides.state().memcpy_calls.size(), 2);
#if CUDA_VERSION >= 12080
  EXPECT_EQ(observation.calls, 0);
#endif
  EXPECT_EQ(overrides.state().memcpy_calls[0].destination, destinations[0]);
  EXPECT_EQ(overrides.state().memcpy_calls[0].source, sources[0]);
  EXPECT_EQ(overrides.state().memcpy_calls[0].size, sizes[0]);
  EXPECT_EQ(overrides.state().memcpy_calls[1].destination, destinations[2]);
  EXPECT_EQ(overrides.state().memcpy_calls[1].source, sources[2]);
  EXPECT_EQ(overrides.state().memcpy_calls[1].size, sizes[2]);
}

TEST(CudaMemcpyBatch, singleton_uses_the_existing_copy_dispatch)
{
  ScopedCudaOverrides overrides;
  CUdeviceptr destinations[]{0x1000};
  CUdeviceptr sources[]{0x2000};
  std::size_t sizes[]{17};

#if CUDA_VERSION >= 12080
  BatchObservation observation;
  overrides.set_batch(make_batch_recorder(observation));
#else
  overrides.reset_batch();
#endif

  EXPECT_EQ(
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 1, CU_STREAM_PER_THREAD),
    CUDA_SUCCESS);
#if CUDA_VERSION >= 12080
  EXPECT_EQ(observation.calls, 1);
  EXPECT_EQ(observation.destinations, std::vector<CUdeviceptr>{destinations[0]});
  EXPECT_EQ(observation.sources, std::vector<CUdeviceptr>{sources[0]});
  EXPECT_EQ(observation.sizes, std::vector<std::size_t>{sizes[0]});
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
#else
  ASSERT_EQ(overrides.state().memcpy_calls.size(), 1);
  EXPECT_EQ(overrides.state().memcpy_calls[0].destination, destinations[0]);
#endif
}

#if CUDA_VERSION >= 12080

TEST(CudaMemcpyBatch, native_batch_receives_all_copies_and_stream_ordering)
{
  ScopedCudaOverrides overrides;
  BatchObservation observation;
  observation.result = CUDA_ERROR_OUT_OF_MEMORY;
  overrides.set_batch(make_batch_recorder(observation));

  CUdeviceptr destinations[]{0x1000, 0x2000, 0x3000};
  CUdeviceptr sources[]{0x4000, 0x5000, 0x6000};
  std::size_t sizes[]{7, 11, 13};
  auto const result =
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 3, CU_STREAM_PER_THREAD);

  EXPECT_EQ(result, observation.result);
  EXPECT_EQ(observation.calls, 1);
  EXPECT_EQ(observation.destinations,
            (std::vector<CUdeviceptr>{destinations[0], destinations[1], destinations[2]}));
  EXPECT_EQ(observation.sources, (std::vector<CUdeviceptr>{sources[0], sources[1], sources[2]}));
  EXPECT_EQ(observation.sizes, (std::vector<std::size_t>{sizes[0], sizes[1], sizes[2]}));
  ASSERT_EQ(observation.attributes.size(), 1);
  EXPECT_EQ(observation.attributes[0].srcAccessOrder, CU_MEMCPY_SRC_ACCESS_ORDER_STREAM);
  EXPECT_EQ(observation.attributes[0].srcLocHint.type, CU_MEM_LOCATION_TYPE_INVALID);
  EXPECT_EQ(observation.attributes[0].srcLocHint.id, 0);
  EXPECT_EQ(observation.attributes[0].dstLocHint.type, CU_MEM_LOCATION_TYPE_INVALID);
  EXPECT_EQ(observation.attributes[0].dstLocHint.id, 0);
  EXPECT_EQ(observation.attributes[0].flags, 0);
  EXPECT_EQ(observation.attribute_indexes, std::vector<std::size_t>{0});
  EXPECT_EQ(observation.stream, CU_STREAM_PER_THREAD);
#if CUDA_VERSION < 13000
  EXPECT_EQ(observation.failure_index, nullptr);
#endif
  EXPECT_TRUE(overrides.state().memcpy_calls.empty());
  EXPECT_TRUE(overrides.state().synchronize_calls.empty());
}

TEST(CudaMemcpyBatch, legacy_default_streams_do_not_use_the_native_batch_api)
{
  ScopedCudaOverrides overrides;
  BatchObservation observation;
  overrides.set_batch(make_batch_recorder(observation));

  CUdeviceptr destinations[]{0x1000, 0x2000};
  CUdeviceptr sources[]{0x3000, 0x4000};
  std::size_t sizes[]{7, 11};

  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 2, nullptr),
            CUDA_SUCCESS);
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_async(destinations[0], sources[0], sizes[0], nullptr),
            CUDA_SUCCESS);
  EXPECT_EQ(
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 2, CU_STREAM_LEGACY),
    CUDA_SUCCESS);
  EXPECT_EQ(
    kvikio::cudaAPI::cuda_memcpy_async(destinations[0], sources[0], sizes[0], CU_STREAM_LEGACY),
    CUDA_SUCCESS);

  EXPECT_EQ(observation.calls, 0);
  ASSERT_EQ(overrides.state().memcpy_calls.size(), 6);
  EXPECT_EQ(overrides.state().memcpy_calls[0].stream, nullptr);
  EXPECT_EQ(overrides.state().memcpy_calls[1].stream, nullptr);
  EXPECT_EQ(overrides.state().memcpy_calls[2].stream, nullptr);
  EXPECT_EQ(overrides.state().memcpy_calls[3].stream, CU_STREAM_LEGACY);
  EXPECT_EQ(overrides.state().memcpy_calls[4].stream, CU_STREAM_LEGACY);
  EXPECT_EQ(overrides.state().memcpy_calls[5].stream, CU_STREAM_LEGACY);
}

#endif

TEST(CudaMemcpyBatch, missing_native_batch_api_uses_the_conventional_copy_api)
{
  ScopedCudaOverrides overrides;
  overrides.reset_batch();
  CUdeviceptr destinations[]{0x1000, 0x2000};
  CUdeviceptr sources[]{0x3000, 0x4000};
  std::size_t sizes[]{7, 11};

  EXPECT_EQ(
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 2, CU_STREAM_PER_THREAD),
    CUDA_SUCCESS);
  ASSERT_EQ(overrides.state().memcpy_calls.size(), 2);
  EXPECT_EQ(overrides.state().memcpy_calls[0].destination, destinations[0]);
  EXPECT_EQ(overrides.state().memcpy_calls[1].destination, destinations[1]);
}

TEST(CudaMemcpyBatch, fallback_stops_at_the_first_error_without_synchronizing)
{
  ScopedCudaOverrides overrides;
  overrides.reset_batch();
  overrides.state().fail_at = 1;
  overrides.state().failure = CUDA_ERROR_OUT_OF_MEMORY;

  CUdeviceptr destinations[]{0x1000, 0x2000, 0x3000};
  CUdeviceptr sources[]{0x4000, 0x5000, 0x6000};
  std::size_t sizes[]{7, 11, 13};
  auto const stream = CU_STREAM_PER_THREAD;
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 3, stream),
            overrides.state().failure);

  ASSERT_EQ(overrides.state().memcpy_calls.size(), 2);
  EXPECT_EQ(overrides.state().memcpy_calls[0].destination, destinations[0]);
  EXPECT_EQ(overrides.state().memcpy_calls[1].destination, destinations[1]);
  EXPECT_TRUE(overrides.state().synchronize_calls.empty());

  // The caller owns lifetime fencing after any enqueue error.
  EXPECT_EQ(kvikio::cudaAPI::instance().StreamSynchronize(stream), CUDA_SUCCESS);
  EXPECT_EQ(overrides.state().synchronize_calls, std::vector<CUstream>{stream});
}

struct CudaHostDeleter {
  void operator()(void* pointer) const noexcept
  {
    if (pointer != nullptr) { std::ignore = cudaFreeHost(pointer); }
  }
};

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

TEST(CudaMemcpyBatch, copies_multiple_independent_buffers_on_a_real_stream)
{
  constexpr std::size_t bytes = 4096;
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));

  void* source_a_raw{};
  void* source_b_raw{};
  void* destination_a_raw{};
  void* destination_b_raw{};
  KVIKIO_CHECK_CUDA(cudaMallocHost(&source_a_raw, bytes));
  std::unique_ptr<void, CudaHostDeleter> source_a{source_a_raw};
  KVIKIO_CHECK_CUDA(cudaMallocHost(&source_b_raw, bytes));
  std::unique_ptr<void, CudaHostDeleter> source_b{source_b_raw};
  KVIKIO_CHECK_CUDA(cudaMalloc(&destination_a_raw, bytes));
  std::unique_ptr<void, CudaDeviceDeleter> destination_a{destination_a_raw};
  KVIKIO_CHECK_CUDA(cudaMalloc(&destination_b_raw, bytes));
  std::unique_ptr<void, CudaDeviceDeleter> destination_b{destination_b_raw};

  std::fill_n(static_cast<std::uint8_t*>(source_a.get()), bytes, std::uint8_t{0x3c});
  std::fill_n(static_cast<std::uint8_t*>(source_b.get()), bytes, std::uint8_t{0xa5});

  cudaStream_t stream_raw{};
  KVIKIO_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_raw, cudaStreamNonBlocking));
  std::unique_ptr<CUstream_st, CudaStreamDeleter> stream{stream_raw};

  CUdeviceptr destinations[]{kvikio::convert_void2deviceptr(destination_a.get()),
                             kvikio::convert_void2deviceptr(destination_b.get())};
  CUdeviceptr sources[]{kvikio::convert_void2deviceptr(source_a.get()),
                        kvikio::convert_void2deviceptr(source_b.get())};
  std::size_t sizes[]{bytes, bytes};
  KVIKIO_CUDA_DRIVER_TRY(
    kvikio::cudaAPI::cuda_memcpy_batch_async(destinations, sources, sizes, 2, stream.get()));
  KVIKIO_CHECK_CUDA(cudaStreamSynchronize(stream.get()));

  std::vector<std::uint8_t> actual_a(bytes);
  std::vector<std::uint8_t> actual_b(bytes);
  KVIKIO_CHECK_CUDA(
    cudaMemcpy(actual_a.data(), destination_a.get(), bytes, cudaMemcpyDeviceToHost));
  KVIKIO_CHECK_CUDA(
    cudaMemcpy(actual_b.data(), destination_b.get(), bytes, cudaMemcpyDeviceToHost));
  EXPECT_TRUE(
    std::all_of(actual_a.begin(), actual_a.end(), [](auto value) { return value == 0x3c; }));
  EXPECT_TRUE(
    std::all_of(actual_b.begin(), actual_b.end(), [](auto value) { return value == 0xa5; }));
}

}  // namespace
