/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {

/**
 * @brief Compute the private bounce-buffer cache cap for one reactor-owned shard.
 *
 * A zero total means unlimited. A finite total must be at least the reactor count. The returned
 * ceiling share is large enough for the busiest reactor when the process-wide request budget is
 * divided exactly across reactors. Request admission still enforces the exact process-wide budget;
 * this value only bounds buffers retained by an individual cache shard.
 *
 * @param max_total Process-wide maximum concurrent request count, or zero for unlimited.
 * @param num_reactors Number of reactors sharing the request budget.
 * @return The per-shard cache cap, or `std::nullopt` for unlimited concurrency.
 * @throws std::invalid_argument if the reactor count is zero or a finite budget is smaller than
 * the reactor count.
 */
[[nodiscard]] KVIKIO_EXPORT std::optional<std::size_t> bounce_buffer_cache_shard_limit(
  std::size_t max_total, std::size_t num_reactors);

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
/**
 * @brief Failure points used to verify exceptional bounce-buffer recycle paths.
 */
enum class BounceBufferCacheFailurePoint { ACCOUNTING_TRANSITION, CALLBACK_INSERTION };

/**
 * @brief Make the next recycle operation throw at @p point.
 *
 * The hook is one-shot and compiled only into test builds.
 */
KVIKIO_EXPORT void inject_bounce_buffer_cache_failure_for_testing(
  BounceBufferCacheFailurePoint point) noexcept;
#endif

/**
 * @brief Per-(thread, CUDA context) cache of bounce buffers with async recycling.
 *
 * Each key of `(std::thread::id, CUcontext)` tracks three counts that together are capped at
 * `*cap` (when `cap` has a value):
 * - A free list of recyclable `Buffer`s
 * - A count of buffers currently checked out to callers (Phase A)
 * - A count of pending CUDA-side recycle callbacks (Phase B).
 *
 * Typical lifecycle of a buffer:
 * - Caller invokes `try_get(ctx)` to acquire a `Buffer` (Phase A begins). The buffer is filled with
 *   host data.
 * - Caller submits a `cuMemcpyAsync` to a `stream` and calls `recycle_after(ctx, buf, stream)`. The
 *   cache schedules a `cuLaunchHostFunc` on the stream that will recycle the buffer to the free
 *   list when the H2D completes (Phase A ends, Phase B begins).
 * - The CUDA driver invokes the recycle callback on a driver thread. The callback moves the Buffer
 *   from "in flight" to "free" (Phase B ends).
 *
 * Failure paths (no H2D submitted) use `recycle_now(ctx, buf)` which returns the buffer to the
 * free list immediately.
 *
 * `try_get` is non-blocking: it returns `std::nullopt` when the cap is reached. Backpressure on
 * the caller side decides whether to retry, defer, or block.
 *
 * @tparam Allocator The allocator policy that determines buffer properties:
 * - PageAlignedAllocator: For host-only Direct I/O
 * - CudaPinnedAllocator: For device I/O without Direct I/O
 * - CudaPageAlignedPinnedAllocator: For device I/O with Direct I/O
 *
 * @note Each in-flight `Buffer` must be checked out and returned by the same `(thread, CUDA
 * context)` key. Mixing threads or contexts within one buffer's lifecycle breaks the per-key
 * accounting.
 */
template <typename Allocator = CudaPinnedAllocator>
class BounceBufferCachePerThreadAndContext {
 public:
  using Buffer = typename BounceBufferPool<Allocator>::Buffer;

  /**
   * @brief Construct a cache with the given per-(thread, CUDA context) buffer cap.
   *
   * @param cap Maximum number of buffers (free + checked-out + in-flight) per `(thread, CUDA
   * context)` key. `std::nullopt` means unlimited, where `try_get` never returns `nullopt`, the
   * cache grows on demand, and backpressure becomes the caller's responsibility.
   */
  explicit BounceBufferCachePerThreadAndContext(std::optional<std::size_t> cap);

  // Non-copyable, non-movable
  BounceBufferCachePerThreadAndContext(BounceBufferCachePerThreadAndContext const&) = delete;
  BounceBufferCachePerThreadAndContext& operator=(BounceBufferCachePerThreadAndContext const&) =
    delete;
  BounceBufferCachePerThreadAndContext(BounceBufferCachePerThreadAndContext&&)            = delete;
  BounceBufferCachePerThreadAndContext& operator=(BounceBufferCachePerThreadAndContext&&) = delete;

  ~BounceBufferCachePerThreadAndContext() = default;

  /**
   * @brief Get the process-wide singleton instance.
   *
   * The instance is constructed lazily on first call with the ceiling of the process-wide request
   * budget divided by the reactor count. This is large enough for the busiest reactor after an
   * exact quotient/remainder split. Request admission, rather than this retained-buffer cache,
   * enforces the exact process-wide concurrency limit. The singleton is intentionally
   * heap-allocated and never deleted. Each template instantiation (different `Allocator`) has its
   * own singleton.
   *
   * @return Reference to the singleton instance.
   */
  KVIKIO_EXPORT static BounceBufferCachePerThreadAndContext& instance();

  /**
   * @brief Get the cap, i.e. the the maximum number of bounce buffers allowed per-(thread, CUDA
   * context), with which this instance was constructed.
   *
   * @return The cap. `std::nullopt` means unlimited.
   */
  [[nodiscard]] std::optional<std::size_t> cap() const noexcept;

  /**
   * @brief Try to acquire a buffer for the given CUDA context (non-blocking).
   *
   * Pops from the per-(this thread, ctx) free list if non-empty. Otherwise allocates a fresh Buffer
   * from the underlying `BounceBufferPool` if the per-key count (`free + checked_out + in_flight`)
   * is below `*cap()` (or `cap()` has no value). When the cap is reached, returns `std::nullopt`.
   *
   * @param ctx The CUDA context to associate the buffer with. The caller is expected to have `ctx`
   * current on the calling thread before calling.
   * @return A `Buffer` on success, `std::nullopt` if the cap is reached.
   *
   * @exception kvikio::CUfileException if the underlying allocation fails.
   */
  [[nodiscard]] std::optional<Buffer> try_get(CUcontext ctx);

  /**
   * @brief Recycle the buffer immediately, without involving CUDA.
   *
   * Used on failure paths where no CUDA work was submitted with this buffer.
   *
   * @param ctx The CUDA context the buffer was acquired under.
   * @param buf The buffer (ownership transferred into the cache).
   */
  void recycle_now(CUcontext ctx, Buffer&& buf);

  /**
   * @brief Schedule the buffer to be recycled when the stream's prior work completes.
   *
   * Records a `cuLaunchHostFunc` on `stream` so that, once `stream` has finished the
   * `cuMemcpyAsync` (or whatever CUDA work) that consumes `buf`, the buffer is returned to the free
   * list on a CUDA driver controlled thread. Non-blocking on the calling thread.
   *
   * @note If callback-side cache insertion fails (for example, an unlimited cache cannot grow its
   * free list under host OOM), the callback detaches the allocation without invoking CUDA,
   * poisons this cache, and fails closed. The allocation is intentionally leaked because CUDA APIs
   * are forbidden from a `cuLaunchHostFunc` callback.
   *
   * @param ctx The CUDA context the buffer was acquired under (must match the original `try_get`
   * call).
   * @param buf The buffer (ownership transferred into the cache).
   * @param stream The CUDA stream whose completion signals that `buf` is safe to recycle.
   * @param on_recycle Optional callable invoked on the CUDA driver thread immediately after the
   * buffer is returned to the free list.
   *
   * @exception kvikio::CUfileException if `cuLaunchHostFunc` fails.
   */
  void recycle_after(CUcontext ctx,
                     Buffer&& buf,
                     CUstream stream,
                     std::function<void()> on_recycle = {});

  /**
   * @brief Quarantine a checked-out buffer after CUDA cannot establish a reuse fence.
   *
   * The allocation is intentionally leaked and the owning shard is poisoned. Its checked-out
   * accounting remains charged so no replacement can hide the lost slot. This function is the
   * terminal recovery path for a failed asynchronous operation followed by a failed stream
   * synchronization.
   *
   * @param ctx The CUDA context under which the buffer was acquired.
   * @param buf The uncertain DMA source. Ownership is consumed without freeing or recycling it.
   */
  void abandon_checked_out_after_failed_sync(CUcontext ctx, Buffer&& buf) noexcept;

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  /**
   * @brief Remove an isolated poisoned shard after a test has freed its deliberately leaked slot.
   *
   * This is test-only cleanup. Production code must never revive a shard after CUDA failed to
   * establish safe buffer reuse.
   */
  void erase_poisoned_shard_for_testing(CUcontext ctx);
#endif

 private:
  struct Shard {
    std::mutex mutex;
    std::vector<Buffer> free;
    std::size_t checked_out{0};
    std::size_t in_flight{0};
    bool poisoned{false};
    // Invariant: !cap.has_value() || free.size() + checked_out + in_flight <= *cap

    explicit Shard(std::optional<std::size_t> cap);
  };

  // Associates a buffer with the shard whose free list will receive it. Heap-allocated and passed
  // as user data to `cuLaunchHostFunc`. The callback deletes this struct after moving the buffer.
  struct RecycleCallbackData {
    BounceBufferCachePerThreadAndContext* owner;
    Shard* shard;
    Buffer buffer;
    // Called after the buffer is returned to the free list.
    std::function<void()> on_recycle;
  };

  static void CUDA_CB recycle_callback(void* user_data);

  /**
   * @brief Retrieve or create the Shard state for the calling thread and given CUDA context.
   *
   * @param ctx The CUDA context the buffer was acquired under.
   */
  Shard& get_shard(CUcontext ctx);

  std::optional<std::size_t> _cap;
  // Emergency cache-wide stop used when a no-throw quarantine path cannot safely make a buffer
  // reusable. This includes callback-side cache insertion failure and inability to reach the
  // shard that necessarily existed when a buffer was checked out.
  std::atomic<bool> _emergency_poisoned{false};
  std::mutex _map_mutex;
  std::map<std::pair<std::thread::id, CUcontext>, std::unique_ptr<Shard>> _shards;
};

}  // namespace kvikio::detail
