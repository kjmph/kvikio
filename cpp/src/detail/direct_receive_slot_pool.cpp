/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>

namespace kvikio::detail {
namespace {

constexpr std::size_t tls_record_size = 16 * 1024;

void log_error_noexcept(char const* message) noexcept
{
  try {
    KVIKIO_LOG_ERROR(message);
  } catch (...) {
  }
}

void validate_configuration(std::size_t slot_size, std::size_t max_pinned_bytes)
{
  auto const minimum = DirectReceiveSlotPool::minimum_slot_size();
  KVIKIO_EXPECT(slot_size != 0, "direct-receive slot size must be nonzero", std::invalid_argument);
  KVIKIO_EXPECT(slot_size >= minimum,
                "direct-receive slot size must be at least " + std::to_string(minimum) + " bytes",
                std::invalid_argument);
  KVIKIO_EXPECT(max_pinned_bytes >= slot_size,
                "direct-receive maximum pinned bytes must be at least one slot",
                std::invalid_argument);

  KVIKIO_EXPECT(max_pinned_bytes / slot_size != 0,
                "direct-receive pinned-byte cap cannot represent one complete slot",
                std::invalid_argument);
}

}  // namespace

struct DirectReceiveSlotPool::State {
  State(std::size_t slot_size_, std::size_t max_pinned_bytes_)
    : slot_size{slot_size_},
      max_pinned_bytes{max_pinned_bytes_},
      max_slots{max_pinned_bytes / slot_size}
  {
  }

  ~State() noexcept
  {
    // Production uses a leaked singleton. This path exists for isolated unit-test pools and for
    // slots that outlive such a pool object. Checked-out slots retain `State`, so only free slots
    // and deliberately quarantined allocations can remain here.
    while (free_head != nullptr) {
      auto* current = free_head;
      free_head     = *static_cast<void**>(free_head);
      try {
        allocator.deallocate(current, slot_size);
      } catch (std::exception const&) {
        log_error_noexcept("direct-receive slot deallocation failed");
      } catch (...) {
        log_error_noexcept("direct-receive slot deallocation failed");
      }
    }
    // Quarantined allocations are intentionally unreachable and remain pinned. Freeing one here
    // could race DMA after CUDA failed to establish a reuse fence.
  }

  void recycle(void* data) noexcept
  {
    if (data == nullptr) { return; }
    try {
      std::lock_guard const lock(mutex);
      if (checked_out_slots == 0) {
        log_error_noexcept("direct-receive slot pool checked-out accounting underflow");
        return;
      }
      --checked_out_slots;
      *static_cast<void**>(data) = free_head;
      free_head                  = data;
      ++free_slots;
    } catch (...) {
      // Slot destruction and CUDA completion callbacks cannot propagate. If bookkeeping itself
      // fails, leave the allocation unreachable rather than risking reuse while its state is
      // uncertain.
      log_error_noexcept("direct-receive slot recycling failed; leaking the allocation");
    }
  }

  void quarantine(void* data)
  {
    if (data == nullptr) { return; }
    std::lock_guard const lock(mutex);
    if (checked_out_slots == 0) {
      KVIKIO_LOG_ERROR("direct-receive slot pool quarantine accounting underflow");
      return;
    }
    --checked_out_slots;
    ++quarantined_slots;
    // Do not link `data` into any owner or free list. The allocation remains included in
    // `allocated_slots`, so a replacement cannot make the pool exceed max_pinned_bytes.
  }

  mutable std::mutex mutex;
  std::size_t const slot_size;
  std::size_t const max_pinned_bytes;
  std::size_t const max_slots;
  CudaPinnedAllocator allocator;
  void* free_head{};
  std::size_t allocating_slots{};
  std::size_t allocated_slots{};
  std::size_t checked_out_slots{};
  std::size_t free_slots{};
  std::size_t quarantined_slots{};
  std::size_t allocation_high_water_slots{};
  std::size_t checked_out_high_water_slots{};
};

DirectReceiveSlotPool::Slot::Slot(std::shared_ptr<State> state, void* data) noexcept
  : _state{std::move(state)}, _data{data}
{
}

DirectReceiveSlotPool::Slot::Slot(Slot&& other) noexcept
  : _state{std::move(other._state)}, _data{std::exchange(other._data, nullptr)}
{
}

DirectReceiveSlotPool::Slot& DirectReceiveSlotPool::Slot::operator=(Slot&& other) noexcept
{
  if (this != std::addressof(other)) {
    reset();
    _state = std::move(other._state);
    _data  = std::exchange(other._data, nullptr);
  }
  return *this;
}

DirectReceiveSlotPool::Slot::~Slot() noexcept { reset(); }

void* DirectReceiveSlotPool::Slot::get() const noexcept { return _data; }

std::size_t DirectReceiveSlotPool::Slot::size() const noexcept
{
  return _state == nullptr ? 0 : _state->slot_size;
}

DirectReceiveSlotPool::Slot::operator bool() const noexcept { return _data != nullptr; }

void DirectReceiveSlotPool::Slot::reset() noexcept
{
  if (_data != nullptr) { _state->recycle(std::exchange(_data, nullptr)); }
  _state.reset();
}

void DirectReceiveSlotPool::Slot::quarantine_after_failed_sync() noexcept
{
  if (_data == nullptr) { return; }
  // Detach the allocation before accounting or logging. A failed CUDA fence means this source
  // must never be recycled, even if defensive bookkeeping itself fails.
  auto* data = std::exchange(_data, nullptr);
  auto state = std::move(_state);
  try {
    state->quarantine(data);
  } catch (...) {
    // `data` is already unreachable and therefore remains safely leaked.
  }
}

void DirectReceiveSlotPool::recycle_completed(std::span<Slot> slots) noexcept
{
  State* owner{};
  for (auto const& slot : slots) {
    if (!slot) { continue; }
    if (owner == nullptr) {
      owner = slot._state.get();
    } else if (owner != slot._state.get()) {
      // Mixed isolated pools are valid in tests and future internal consumers. Preserve ownership
      // by using each slot's ordinary self-contained return path.
      for (auto& mixed_slot : slots) {
        mixed_slot.reset();
      }
      return;
    }
  }
  if (owner == nullptr) { return; }

  auto state = [&slots] {
    for (auto const& slot : slots) {
      if (slot) { return slot._state; }
    }
    return std::shared_ptr<State>{};
  }();

  void* head{};
  void* tail{};
  std::size_t count{};
  for (auto& slot : slots) {
    if (!slot) { continue; }
    auto* data = std::exchange(slot._data, nullptr);
    slot._state.reset();
    *static_cast<void**>(data) = head;
    head                       = data;
    if (tail == nullptr) { tail = data; }
    ++count;
  }

  try {
    std::lock_guard const lock(state->mutex);
    if (state->checked_out_slots < count) {
      log_error_noexcept("direct-receive slot pool bulk accounting underflow; leaking slots");
      return;
    }
    *static_cast<void**>(tail) = state->free_head;
    state->free_head           = head;
    state->checked_out_slots -= count;
    state->free_slots += count;
  } catch (...) {
    // Every slot was detached before bookkeeping. If the lock or accounting path fails, the
    // allocations remain unreachable and capacity-charged instead of becoming unsafely reusable.
    log_error_noexcept("direct-receive bulk slot recycling failed; leaking the allocations");
  }
}

DirectReceiveSlotPool::DirectReceiveSlotPool(std::size_t slot_size, std::size_t max_pinned_bytes)
{
  validate_configuration(slot_size, max_pinned_bytes);
  _state = std::make_shared<State>(slot_size, max_pinned_bytes);
}

DirectReceiveSlotPool& DirectReceiveSlotPool::instance()
{
  static auto* pool = [] {
    auto const [slot_size, max_pinned_bytes] =
      defaults::freeze_remote_direct_receive_slot_pool_configuration();
    return new DirectReceiveSlotPool(slot_size, max_pinned_bytes);
  }();
  return *pool;
}

std::size_t DirectReceiveSlotPool::minimum_slot_size() noexcept
{
  return std::max<std::size_t>(tls_record_size, CURL_MAX_WRITE_SIZE);
}

std::optional<DirectReceiveSlotPool::Slot> DirectReceiveSlotPool::try_acquire()
{
  void* data{};
  {
    std::lock_guard const lock(_state->mutex);
    if (_state->free_head != nullptr) {
      data              = _state->free_head;
      _state->free_head = *static_cast<void**>(_state->free_head);
      --_state->free_slots;
      ++_state->checked_out_slots;
      _state->checked_out_high_water_slots =
        std::max(_state->checked_out_high_water_slots, _state->checked_out_slots);
      return Slot{_state, data};
    }
    if (_state->allocated_slots + _state->allocating_slots >= _state->max_slots) {
      return std::nullopt;
    }

    // Reserve the slot under the lock before allocating outside it. Concurrent callers therefore
    // cannot collectively overshoot the KvikIO-wide byte cap.
    ++_state->allocating_slots;
  }

  try {
    data = _state->allocator.allocate(_state->slot_size);
  } catch (...) {
    std::lock_guard const lock(_state->mutex);
    --_state->allocating_slots;
    throw;
  }
  {
    std::lock_guard const lock(_state->mutex);
    --_state->allocating_slots;
    ++_state->allocated_slots;
    ++_state->checked_out_slots;
    _state->allocation_high_water_slots =
      std::max(_state->allocation_high_water_slots, _state->allocated_slots);
    _state->checked_out_high_water_slots =
      std::max(_state->checked_out_high_water_slots, _state->checked_out_slots);
  }
  return Slot{_state, data};
}

void DirectReceiveSlotPool::quarantine_after_failed_sync(Slot&& slot) noexcept
{
  if (!slot) { return; }
  auto const foreign_pool = slot._state.get() != _state.get();
  slot.quarantine_after_failed_sync();
  if (foreign_pool) {
    log_error_noexcept("quarantined a direct-receive slot owned by another receive pool");
  }
}

DirectReceiveSlotPoolSnapshot DirectReceiveSlotPool::snapshot() const
{
  std::lock_guard const lock(_state->mutex);
  return {.slot_size                    = _state->slot_size,
          .max_pinned_bytes             = _state->max_pinned_bytes,
          .max_slots                    = _state->max_slots,
          .allocating_slots             = _state->allocating_slots,
          .allocated_slots              = _state->allocated_slots,
          .checked_out_slots            = _state->checked_out_slots,
          .free_slots                   = _state->free_slots,
          .quarantined_slots            = _state->quarantined_slots,
          .allocation_high_water_slots  = _state->allocation_high_water_slots,
          .checked_out_high_water_slots = _state->checked_out_high_water_slots};
}

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
std::unique_ptr<DirectReceiveSlotPool> DirectReceiveSlotPool::create_for_testing(
  std::size_t slot_size, std::size_t max_pinned_bytes)
{
  return std::unique_ptr<DirectReceiveSlotPool>(
    new DirectReceiveSlotPool(slot_size, max_pinned_bytes));
}
#endif

}  // namespace kvikio::detail
