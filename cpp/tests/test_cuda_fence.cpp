/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstddef>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <kvikio/detail/cuda_fence.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace {

std::atomic<std::size_t> context_synchronize_calls{};
decltype(kvikio::cudaAPI::instance().CtxSynchronize) saved_context_synchronize{};

CUresult CUDAAPI count_and_forward_context_synchronize()
{
  context_synchronize_calls.fetch_add(1, std::memory_order_relaxed);
  return saved_context_synchronize();
}

CUresult CUDAAPI fail_context_synchronize() { return CUDA_ERROR_INVALID_CONTEXT; }

class ScopedContextSynchronizeOverride {
 public:
  explicit ScopedContextSynchronizeOverride(
    decltype(kvikio::cudaAPI::instance().CtxSynchronize) replacement)
  {
    auto& slot                = kvikio::cudaAPI::instance().CtxSynchronize;
    saved_context_synchronize = slot;
    context_synchronize_calls = 0;
    slot                      = replacement;
  }

  ScopedContextSynchronizeOverride(ScopedContextSynchronizeOverride const&)            = delete;
  ScopedContextSynchronizeOverride& operator=(ScopedContextSynchronizeOverride const&) = delete;

  ~ScopedContextSynchronizeOverride()
  {
    kvikio::cudaAPI::instance().CtxSynchronize = saved_context_synchronize;
  }
};

CUcontext current_context()
{
  CUcontext context{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetCurrent(&context));
  return context;
}

class CudaFenceTest : public testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }
};

TEST_F(CudaFenceTest, success_does_not_fence_context)
{
  ScopedContextSynchronizeOverride const override{&count_and_forward_context_synchronize};
  EXPECT_NO_THROW(kvikio::detail::run_with_context_fence_on_failure(current_context(), [] {}));
  EXPECT_EQ(context_synchronize_calls.load(std::memory_order_relaxed), 0);
}

TEST_F(CudaFenceTest, operation_failure_fences_context_then_preserves_primary_error)
{
  ScopedContextSynchronizeOverride const override{&count_and_forward_context_synchronize};
  EXPECT_THROW(
    {
      try {
        kvikio::detail::run_with_context_fence_on_failure(
          current_context(), [] { throw std::runtime_error{"injected CUDA completion failure"}; });
      } catch (std::runtime_error const& error) {
        EXPECT_STREQ(error.what(), "injected CUDA completion failure");
        throw;
      }
    },
    std::runtime_error);
  EXPECT_EQ(context_synchronize_calls.load(std::memory_order_relaxed), 1);
}

TEST_F(CudaFenceTest, failed_context_fence_terminates)
{
  auto const context = current_context();
  EXPECT_DEATH(
    {
      kvikio::cudaAPI::instance().CtxSynchronize = &fail_context_synchronize;
      kvikio::detail::run_with_context_fence_on_failure(
        context, [] { throw std::runtime_error{"injected CUDA completion failure"}; });
    },
    "");
}

}  // namespace
