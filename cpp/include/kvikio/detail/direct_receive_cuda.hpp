/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>

#include <kvikio/detail/direct_receive.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/detail/event.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio::detail {

/**
 * @brief Maximum independent copies and source slots in one direct-receive CUDA submission.
 *
 * Matching the receive descriptor's fixed span bound lets one pathological slot remain atomic
 * while bounding all submission bookkeeping. Callers flush a full batch before adding more work.
 */
inline constexpr std::size_t direct_receive_max_copies_per_cuda_batch =
  direct_receive_max_spans_per_slot;
inline constexpr std::size_t direct_receive_max_slots_per_cuda_batch =
  direct_receive_max_spans_per_slot;

enum class DirectReceiveCudaAddResult : std::uint8_t {
  added,
  batch_full,
  batch_incompatible,
};

/** @brief Receive path whose KvikIO H2D batch submission this object accounts for. */
enum class DirectReceiveCudaPath : std::uint8_t {
  strict_rx,
  copied_stream,
};

struct DirectReceiveCudaSubmission {
  std::size_t slots{};
  std::size_t copies{};
  std::size_t bytes{};
  bool asynchronous{};
};

/**
 * @brief Fixed-capacity owner of one pollable pinned-host-to-device copy batch.
 *
 * Released receive slots are flattened into independent CUDA copy descriptors without retaining
 * or copying their large scatter descriptions. Slots remain owned until a completion event proves
 * that CUDA no longer reads them. Multiple logical reads may share one CUDA batch only when
 * every destination range is provably disjoint and every completion barrier belongs to the same
 * CUDA context.
 *
 * One KvikIO batch submission may contain work from independent logical reads. CUDA cannot
 * identify which descriptor caused a submission or completion failure, so such a failure is
 * deliberately reported to every logical owner represented in that batch.
 *
 * All validation happens before slot ownership changes. All completion events and barrier state
 * are prepared before the first H2D call. Once submission begins, an error synchronizes the stream
 * before recycling any slot; if that fence fails, sources are permanently quarantined and every
 * affected destination barrier is marked for a context-wide fallback fence.
 */
class KVIKIO_EXPORT DirectReceiveCudaBatch {
 public:
  /**
   * @brief Construct an empty batch for one context, stream, and accounting path.
   *
   * `stream` must remain valid through terminal completion or destruction of this object.
   */
  DirectReceiveCudaBatch(CUcontext cuda_context, CUstream stream, DirectReceiveCudaPath path);
  ~DirectReceiveCudaBatch() noexcept;

  DirectReceiveCudaBatch(DirectReceiveCudaBatch const&)            = delete;
  DirectReceiveCudaBatch& operator=(DirectReceiveCudaBatch const&) = delete;
  DirectReceiveCudaBatch(DirectReceiveCudaBatch&&)                 = delete;
  DirectReceiveCudaBatch& operator=(DirectReceiveCudaBatch&&)      = delete;

  /**
   * @brief Add one released slot if the fixed batch has capacity.
   *
   * @param slot Slot holding the source bytes. It is moved from only when `added` is returned.
   * @param released Validated receive description produced for this slot.
   * @param destination_base Base address of the complete requested destination range.
   * @param destination_extent Size of that complete destination range.
   * @param barrier Lifetime barrier for the logical read receiving these bytes. The aggregate
   * completion path must retain an independent owner until every terminal cookie has been reported
   * and it has synchronized the barrier, including after a failed local stream fence.
   * @param completion_cookie Opaque value returned after this slot's H2D completes.
   * @return `added`; `batch_full` when a fixed capacity was reached; or `batch_incompatible` when
   * the work is valid by itself but overlaps a destination owned by a different logical read in
   * this batch. Neither non-added result changes any argument or batch state.
   * @throws std::invalid_argument for malformed, same-logical-read overlapping, internally
   * overlapping, out-of-range, or wrong-context work.
   * @throws std::logic_error unless the batch is still collecting work.
   */
  [[nodiscard]] DirectReceiveCudaAddResult try_add(DirectReceiveSlotPool::Slot& slot,
                                                   DirectReceiveReleasedSlot const& released,
                                                   CUdeviceptr destination_base,
                                                   std::size_t destination_extent,
                                                   std::shared_ptr<IoEventBarrier> barrier,
                                                   std::uintptr_t completion_cookie);

  /**
   * @brief Submit all collected copies and record their completion fences.
   *
   * Header-only slots complete synchronously when a batch contains no body spans.
   * Failures before the first H2D attempt leave the batch collecting and retryable. Failures after
   * an H2D may have been enqueued leave it terminally failed after recycling or quarantining every
   * source slot.
   */
  [[nodiscard]] DirectReceiveCudaSubmission submit();

  /**
   * @brief Poll CUDA once, recycling all sources when the completion event is ready.
   *
   * @return true after completion has been established, false while work remains in flight.
   */
  [[nodiscard]] bool poll();

  /**
   * @brief Wait for CUDA completion and recycle all source slots.
   */
  void wait();

  /**
   * @brief Opaque cookies for slots whose batch reached a terminal state.
   *
   * Cookies become available after successful completion or after a post-enqueue/completion error
   * has safely recycled or quarantined every source slot. Duplicate values represent one terminal
   * occurrence per slot. The returned view remains valid only until reset or destruction.
   *
   * A failed cookie proves source-slot safety, not destination safety. When the local stream fence
   * also failed, the aggregate completion owner must synchronize its barrier before exposing the
   * failure or releasing the destination.
   */
  [[nodiscard]] std::span<std::uintptr_t const> finished_cookies() const;

  /**
   * @brief Clear a collecting, completed, or failed batch for reuse.
   *
   * A submitted batch must first complete through `poll()` or `wait()`.
   */
  void reset();

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool submitted() const noexcept;
  [[nodiscard]] bool completed() const noexcept;
  [[nodiscard]] bool failed() const noexcept;
  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t copy_count() const noexcept;
  [[nodiscard]] std::size_t body_bytes() const noexcept;

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  /**
   * @brief Make the next submission fail before allocating or enqueueing CUDA work.
   */
  static void inject_pre_submit_allocation_failure_for_testing() noexcept;

  /**
   * @brief Make the next submission fail after its H2D batch has been enqueued.
   */
  static void inject_post_enqueue_failure_for_testing() noexcept;
#endif

 private:
  enum class State : std::uint8_t {
    collecting,
    submitted,
    completed,
    failed,
  };

  void complete_successfully() noexcept;
  [[noreturn]] void recover_after_cuda_failure(std::exception_ptr primary_error);
  void fail_closed_noexcept() noexcept;
  void recycle_slots() noexcept;
  void quarantine_slots() noexcept;
  void mark_barriers_unknown() noexcept;
  void clear_metadata() noexcept;

  CUcontext _cuda_context;
  CUstream _stream;
  DirectReceiveCudaPath _path;
  State _state{State::collecting};
  std::size_t _slot_count{};
  std::size_t _copy_count{};
  std::size_t _barrier_count{};
  std::size_t _body_bytes{};
  std::array<DirectReceiveSlotPool::Slot, direct_receive_max_slots_per_cuda_batch> _slots{};
  std::array<std::uintptr_t, direct_receive_max_slots_per_cuda_batch> _cookies{};
  std::array<CUdeviceptr, direct_receive_max_copies_per_cuda_batch> _destinations{};
  std::array<CUdeviceptr, direct_receive_max_copies_per_cuda_batch> _sources{};
  std::array<std::size_t, direct_receive_max_copies_per_cuda_batch> _sizes{};
  std::array<IoEventBarrier*, direct_receive_max_copies_per_cuda_batch> _copy_barriers{};
  std::array<std::shared_ptr<IoEventBarrier>, direct_receive_max_slots_per_cuda_batch> _barriers{};
  std::optional<CudaEventPool::CudaEvent> _completion_event;
};

}  // namespace kvikio::detail
