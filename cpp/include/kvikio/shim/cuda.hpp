/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <any>
#include <functional>

#include <cuda.h>
#include <kvikio/shim/utils.hpp>
#include <stdexcept>

namespace kvikio {

namespace detail {
/**
 * @brief Non-templated class to hold any callable that returns CUresult
 */
class AnyCallable {
 private:
  std::any _callable;

 public:
  /**
   * @brief Assign a callable to the object
   *
   * @tparam Callable A callable that must return CUresult
   * @param c The callable object
   */
  template <typename Callable>
  void set(Callable&& c)
  {
    _callable = std::function(c);
  }

  /**
   * @brief Destroy the contained callable
   */
  void reset() { _callable.reset(); }

  /**
   * @brief Invoke the container callable
   *
   * @tparam Args Types of the argument. Must exactly match the parameter types of the contained
   * callable. For example, if the parameter is `std::size_t*`, an argument of `nullptr` must be
   * explicitly cast to `std::size_t*`.
   * @param args Arguments to be passed
   * @return CUDA driver API error code
   * @exception std::bad_any_cast if any argument type does not exactly match the parameter type of
   * the contained callable.
   */
  template <typename... Args>
  CUresult operator()(Args... args)
  {
    using T = std::function<CUresult(Args...)>;
    if (!_callable.has_value()) {
      throw std::runtime_error("No callable has been assigned to the wrapper yet.");
    }
    return std::any_cast<T&>(_callable)(args...);
  }

  /**
   * @brief Check if the object holds a callable
   */
  operator bool() const { return _callable.has_value(); }
};

}  // namespace detail

/**
 * @brief Shim layer of the cuda C-API
 *
 * This is a singleton class that use `dlopen` on construction to load the C-API of cuda.
 *
 * For example, `cudaAPI::instance().MemHostAlloc()` corresponds to calling `cuMemHostAlloc()`
 */
class cudaAPI {
 public:
  int driver_version{0};

  decltype(cuInit)* Init{nullptr};
  decltype(cuMemHostAlloc)* MemHostAlloc{nullptr};
  decltype(cuMemFreeHost)* MemFreeHost{nullptr};
  decltype(cuMemHostRegister)* MemHostRegister{nullptr};
  decltype(cuMemHostUnregister)* MemHostUnregister{nullptr};
  decltype(cuMemcpyHtoDAsync)* MemcpyHtoDAsync{nullptr};
  decltype(cuMemcpyDtoHAsync)* MemcpyDtoHAsync{nullptr};
  decltype(cuMemcpyAsync)* MemcpyAsync{nullptr};

  detail::AnyCallable MemcpyBatchAsync{};

  decltype(cuPointerGetAttribute)* PointerGetAttribute{nullptr};
  decltype(cuPointerGetAttributes)* PointerGetAttributes{nullptr};
  decltype(cuCtxCreate)* CtxCreate{nullptr};
  decltype(cuCtxDestroy)* CtxDestroy{nullptr};
  decltype(cuCtxPushCurrent)* CtxPushCurrent{nullptr};
  decltype(cuCtxPopCurrent)* CtxPopCurrent{nullptr};
  decltype(cuCtxGetCurrent)* CtxGetCurrent{nullptr};
  decltype(cuCtxSynchronize)* CtxSynchronize{nullptr};
  decltype(cuCtxGetDevice)* CtxGetDevice{nullptr};
  decltype(cuMemGetAddressRange)* MemGetAddressRange{nullptr};
  decltype(cuGetErrorName)* GetErrorName{nullptr};
  decltype(cuGetErrorString)* GetErrorString{nullptr};
  decltype(cuDeviceGet)* DeviceGet{nullptr};
  decltype(cuDeviceGetCount)* DeviceGetCount{nullptr};
  decltype(cuDeviceGetAttribute)* DeviceGetAttribute{nullptr};
  decltype(cuDevicePrimaryCtxRetain)* DevicePrimaryCtxRetain{nullptr};
  decltype(cuDevicePrimaryCtxRelease)* DevicePrimaryCtxRelease{nullptr};
  decltype(cuStreamSynchronize)* StreamSynchronize{nullptr};
  decltype(cuStreamCreate)* StreamCreate{nullptr};
  decltype(cuStreamDestroy)* StreamDestroy{nullptr};
  decltype(cuDriverGetVersion)* DriverGetVersion{nullptr};
  decltype(cuEventSynchronize)* EventSynchronize{nullptr};
  decltype(cuEventCreate)* EventCreate{nullptr};
  decltype(cuEventDestroy)* EventDestroy{nullptr};
  decltype(cuEventRecord)* EventRecord{nullptr};
  decltype(cuEventQuery)* EventQuery{nullptr};
  decltype(cuLaunchHostFunc)* LaunchHostFunc{nullptr};

 private:
  cudaAPI();

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  KVIKIO_EXPORT static cudaAPI& instance();

  /**
   * @brief Asynchronous memcpy that prefers `cuMemcpyBatchAsync` when supported.
   *
   * Dispatches to `cuMemcpyBatchAsync` with `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`
   * on CUDA >= 12.8 when `stream` is not a legacy default stream; otherwise falls back to
   * `cuMemcpyAsync`. The fallback is mandatory on the legacy default stream, which
   * `cuMemcpyBatchAsync` rejects.
   *
   * @param dst    Destination pointer (host or device under UVA).
   * @param src    Source pointer (host or device under UVA).
   * @param size   Number of bytes to copy.
   * @param stream CUDA stream for ordering.
   * @return CUresult from the underlying driver call.
   */
  static CUresult cuda_memcpy_async(CUdeviceptr dst,
                                    CUdeviceptr src,
                                    std::size_t size,
                                    CUstream stream);

  /**
   * @brief Enqueue multiple independent copies as one CUDA batch when available.
   *
   * CUDA 12.8+ uses `cuMemcpyBatchAsync`; older drivers/toolkits and legacy default streams use a
   * loop of `cuMemcpyAsync`. Zero-sized copies are ignored. Copies within a native CUDA batch have
   * no ordering guarantees and must be independent; overlapping destinations or dependencies
   * between copies result in undefined behavior.
   *
   * This function only enqueues work and never synchronizes. On success, source and destination
   * allocations must remain alive until the stream completes. On failure, the caller must assume
   * that any of the copies may have been enqueued and must successfully fence the stream or context
   * before releasing or reusing any source or destination allocation. The descriptor arrays only
   * need to remain alive until this function returns.
   *
   * @param dsts   Array of destination pointers. Must be non-null when `count` is nonzero.
   * @param srcs   Array of source pointers. Must be non-null when `count` is nonzero.
   * @param sizes  Array of copy sizes. Must be non-null when `count` is nonzero.
   * @param count  Number of entries in `dsts`, `srcs`, and `sizes`.
   * @param stream CUDA stream for ordering.
   * @return `CUDA_SUCCESS`, or the error returned by the underlying CUDA operation.
   */
  static CUresult cuda_memcpy_batch_async(
    CUdeviceptr* dsts, CUdeviceptr* srcs, std::size_t* sizes, std::size_t count, CUstream stream);
};

/**
 * @brief Check if the CUDA library is available
 *
 * Notice, this doesn't check if the runtime environment supports CUDA.
 *
 * @return The boolean answer
 */
bool is_cuda_available();

}  // namespace kvikio
