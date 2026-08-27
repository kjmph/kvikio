/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include <kvikio/shim/utils.hpp>

namespace kvikio::detail {

/**
 * @brief Point-in-time accounting for a direct-receive pinned-slot pool.
 *
 * `allocated_slots` includes quarantined allocations: an allocation whose reuse cannot be proven
 * safe continues to consume the configured byte budget for the lifetime of the process.
 */
struct DirectReceiveSlotPoolSnapshot {
  std::size_t slot_size{};
  std::size_t max_pinned_bytes{};
  std::size_t max_slots{};
  std::size_t allocating_slots{};
  std::size_t allocated_slots{};
  std::size_t checked_out_slots{};
  std::size_t free_slots{};
  std::size_t quarantined_slots{};
  std::size_t allocation_high_water_slots{};
  std::size_t checked_out_high_water_slots{};

  [[nodiscard]] std::size_t allocated_bytes() const noexcept { return allocated_slots * slot_size; }

  [[nodiscard]] std::size_t reserved_bytes() const noexcept
  {
    return (allocated_slots + allocating_slots) * slot_size;
  }
};

/**
 * @brief Bounded KvikIO-wide pool of portable CUDA-pinned direct-receive slots.
 *
 * The production singleton captures the two direct-receive pool defaults when it is first
 * obtained. Slots are allocated lazily, so initializing the pool does not pin host memory. The
 * pool owns an exact configured ceiling for requested slot bytes, independent of generic
 * bounce-buffer size, remote task size, CUDA-context count, and HTTP concurrency.
 *
 * Free-list operations do not allocate. This lets a CUDA completion callback return a slot without
 * allocating memory or invoking a CUDA API. `try_acquire()` may allocate and therefore must run on
 * an ordinary KvikIO/reactor thread with the destination CUDA context current.
 */
class DirectReceiveSlotPool {
 private:
  struct State;

 public:
  /**
   * @brief Move-only ownership of one pinned receive slot.
   */
  class Slot {
   public:
    Slot() noexcept              = default;
    Slot(Slot const&)            = delete;
    Slot& operator=(Slot const&) = delete;
    Slot(Slot&& other) noexcept;
    Slot& operator=(Slot&& other) noexcept;
    ~Slot() noexcept;

    [[nodiscard]] void* get() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

    /**
     * @brief Return this slot to the pool immediately.
     *
     * The caller must already have established that no asynchronous user can access the slot.
     */
    void reset() noexcept;

   private:
    friend class DirectReceiveSlotPool;
    Slot(std::shared_ptr<State> state, void* data) noexcept;

    std::shared_ptr<State> _state;
    void* _data{};
  };

  DirectReceiveSlotPool(DirectReceiveSlotPool const&)            = delete;
  DirectReceiveSlotPool& operator=(DirectReceiveSlotPool const&) = delete;
  DirectReceiveSlotPool(DirectReceiveSlotPool&&)                 = delete;
  DirectReceiveSlotPool& operator=(DirectReceiveSlotPool&&)      = delete;
  ~DirectReceiveSlotPool() noexcept                              = default;

  /**
   * @brief Return the production pool for this loaded KvikIO library.
   *
   * The first call atomically freezes the configured slot size and pinned-byte cap. The singleton
   * is intentionally leaked to avoid CUDA calls during static destruction.
   */
  KVIKIO_EXPORT static DirectReceiveSlotPool& instance();

  /**
   * @brief Configured slot-size floor chosen to accommodate TLS records and write callbacks.
   */
  [[nodiscard]] KVIKIO_EXPORT static std::size_t minimum_slot_size() noexcept;

  /**
   * @brief Try to acquire one slot without waiting.
   *
   * Reuses an allocation when possible, otherwise lazily allocates one without exceeding the
   * configured byte ceiling. Returns `std::nullopt` when every permitted slot is checked out,
   * currently being allocated, or quarantined. Callers that require eventual progress must treat
   * permanent quarantine exhaustion as terminal rather than retrying indefinitely.
   */
  [[nodiscard]] KVIKIO_EXPORT std::optional<Slot> try_acquire();

  /**
   * @brief Permanently quarantine a slot whose asynchronous reuse fence could not be established.
   *
   * The underlying CUDA-pinned allocation is intentionally leaked, remains charged against the
   * KvikIO-wide cap, and can never be returned by a subsequent `try_acquire()`.
   */
  KVIKIO_EXPORT void quarantine_after_failed_sync(Slot&& slot) noexcept;

  /**
   * @brief Snapshot current pool configuration and accounting.
   */
  [[nodiscard]] KVIKIO_EXPORT DirectReceiveSlotPoolSnapshot snapshot() const;

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  /**
   * @brief Construct an isolated pool without freezing production defaults (tests only).
   */
  [[nodiscard]] static std::unique_ptr<DirectReceiveSlotPool> create_for_testing(
    std::size_t slot_size, std::size_t max_pinned_bytes);
#endif

 private:
  DirectReceiveSlotPool(std::size_t slot_size, std::size_t max_pinned_bytes);

  std::shared_ptr<State> _state;
};

}  // namespace kvikio::detail
