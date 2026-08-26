/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <exception>
#include <utility>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

/**
 * @brief Establishes completion of all work in a CUDA context or terminates.
 *
 * This is a last-resort lifetime fence. If it fails, returning would allow caller-owned source or
 * destination storage to unwind while CUDA may still access it.
 */
inline void context_fence_or_terminate(CUcontext context) noexcept
{
  try {
    PushAndPopContext current{context};
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxSynchronize());
  } catch (...) {
    std::terminate();
  }
}

/**
 * @brief Runs CUDA enqueue/completion work with a context-wide lifetime fallback.
 *
 * If @p operation throws, this function first proves that all context work has completed and then
 * rethrows the original exception. A failed fallback fence terminates the process.
 */
template <typename Operation>
void run_with_context_fence_on_failure(CUcontext context, Operation&& operation)
{
  try {
    std::forward<Operation>(operation)();
  } catch (...) {
    auto const primary_error = std::current_exception();
    context_fence_or_terminate(context);
    std::rethrow_exception(primary_error);
  }
}

}  // namespace kvikio::detail
