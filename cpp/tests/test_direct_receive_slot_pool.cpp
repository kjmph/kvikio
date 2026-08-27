/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

#include "utils/utils.hpp"

namespace {

using kvikio::detail::DirectReceiveSlotPool;

CUresult CUDAAPI fail_mem_host_alloc(void**, std::size_t, unsigned int)
{
  return CUDA_ERROR_OUT_OF_MEMORY;
}

class ScopedMemHostAllocOverride {
 public:
  using Function = decltype(kvikio::cudaAPI::instance().MemHostAlloc);

  explicit ScopedMemHostAllocOverride(Function replacement)
    : _slot{kvikio::cudaAPI::instance().MemHostAlloc}, _saved{_slot}
  {
    _slot = replacement;
  }

  ScopedMemHostAllocOverride(ScopedMemHostAllocOverride const&)            = delete;
  ScopedMemHostAllocOverride& operator=(ScopedMemHostAllocOverride const&) = delete;

  ~ScopedMemHostAllocOverride() { _slot = _saved; }

 private:
  Function& _slot;
  Function _saved;
};

class DirectReceiveSlotPoolTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
};

TEST_F(DirectReceiveSlotPoolTest, RejectsInvalidConfiguration)
{
  auto const minimum = DirectReceiveSlotPool::minimum_slot_size();
  EXPECT_GE(minimum, 16 * 1024);

  EXPECT_THROW(DirectReceiveSlotPool::create_for_testing(minimum - 1, minimum),
               std::invalid_argument);
  EXPECT_THROW(DirectReceiveSlotPool::create_for_testing(minimum, minimum - 1),
               std::invalid_argument);
}

TEST_F(DirectReceiveSlotPoolTest, AllocatesLazilyAndEnforcesExactByteCeiling)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool = DirectReceiveSlotPool::create_for_testing(slot_size, 2 * slot_size + slot_size / 2);

  auto initial = pool->snapshot();
  EXPECT_EQ(initial.max_slots, 2);
  EXPECT_EQ(initial.allocating_slots, 0);
  EXPECT_EQ(initial.allocated_slots, 0);
  EXPECT_EQ(initial.allocated_bytes(), 0);

  auto first  = pool->try_acquire();
  auto second = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_NE(first->get(), second->get());
  EXPECT_FALSE(pool->try_acquire().has_value());

  auto full = pool->snapshot();
  EXPECT_EQ(full.allocated_slots, 2);
  EXPECT_EQ(full.checked_out_slots, 2);
  EXPECT_EQ(full.allocated_bytes(), 2 * slot_size);
  EXPECT_EQ(full.reserved_bytes(), 2 * slot_size);
  EXPECT_EQ(full.free_slots, 0);
  EXPECT_EQ(full.quarantined_slots, 0);
  EXPECT_EQ(full.allocation_high_water_slots, 2);
  EXPECT_EQ(full.checked_out_high_water_slots, 2);
  EXPECT_LE(full.allocated_bytes(), full.max_pinned_bytes);

  auto* first_address = first->get();
  first->reset();
  auto after_release = pool->snapshot();
  EXPECT_EQ(after_release.allocated_slots, 2);
  EXPECT_EQ(after_release.checked_out_slots, 1);
  EXPECT_EQ(after_release.free_slots, 1);
  auto reused = pool->try_acquire();
  ASSERT_TRUE(reused.has_value());
  EXPECT_EQ(reused->get(), first_address);
  auto after_reuse = pool->snapshot();
  EXPECT_EQ(after_reuse.allocated_slots, 2);
  EXPECT_EQ(after_reuse.checked_out_slots, 2);
  EXPECT_EQ(after_reuse.free_slots, 0);
  EXPECT_EQ(after_reuse.allocation_high_water_slots, 2);
  EXPECT_EQ(after_reuse.checked_out_high_water_slots, 2);
}

TEST_F(DirectReceiveSlotPoolTest, AllocationFailureReleasesTheReservedCapacity)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool            = DirectReceiveSlotPool::create_for_testing(slot_size, slot_size);

  {
    ScopedMemHostAllocOverride const override{&fail_mem_host_alloc};
    EXPECT_THROW((void)pool->try_acquire(), kvikio::CUfileException);
  }

  auto after_failure = pool->snapshot();
  EXPECT_EQ(after_failure.allocating_slots, 0);
  EXPECT_EQ(after_failure.allocated_slots, 0);
  EXPECT_EQ(after_failure.checked_out_slots, 0);
  EXPECT_EQ(after_failure.reserved_bytes(), 0);

  auto recovered = pool->try_acquire();
  ASSERT_TRUE(recovered.has_value());
  EXPECT_NE(recovered->get(), nullptr);
}

TEST_F(DirectReceiveSlotPoolTest, MoveAssignmentReturnsPreviouslyOwnedSlot)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool            = DirectReceiveSlotPool::create_for_testing(slot_size, 2 * slot_size);
  auto first           = pool->try_acquire();
  auto second          = pool->try_acquire();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  auto* first_address = first->get();

  *first = std::move(*second);
  EXPECT_FALSE(static_cast<bool>(*second));
  auto recycled = pool->try_acquire();
  ASSERT_TRUE(recycled.has_value());
  EXPECT_EQ(recycled->get(), first_address);
}

TEST_F(DirectReceiveSlotPoolTest, CapIsPoolWideAcrossThreads)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool            = DirectReceiveSlotPool::create_for_testing(slot_size, 4 * slot_size);

  std::vector<DirectReceiveSlotPool::Slot> held;
  held.reserve(4);
  std::mutex held_mutex;
  std::vector<std::thread> threads;
  threads.reserve(12);
  for (int i = 0; i < 12; ++i) {
    threads.emplace_back([&] {
      KVIKIO_CHECK_CUDA(cudaSetDevice(0));
      auto slot = pool->try_acquire();
      if (!slot.has_value()) { return; }
      std::lock_guard const lock(held_mutex);
      held.push_back(std::move(slot.value()));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(held.size(), 4);
  EXPECT_FALSE(pool->try_acquire().has_value());
  auto const snapshot = pool->snapshot();
  EXPECT_EQ(snapshot.allocated_slots, 4);
  EXPECT_EQ(snapshot.checked_out_slots, 4);
  EXPECT_EQ(snapshot.allocation_high_water_slots, 4);
  EXPECT_EQ(snapshot.checked_out_high_water_slots, 4);
}

TEST_F(DirectReceiveSlotPoolTest, ConcurrentAcquireAndRecyclePreservesAccounting)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool            = DirectReceiveSlotPool::create_for_testing(slot_size, 4 * slot_size);
  constexpr std::size_t thread_count{12};
  constexpr std::size_t iterations{500};
  std::atomic<std::size_t> completed{};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&] {
      KVIKIO_CHECK_CUDA(cudaSetDevice(0));
      for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        std::optional<DirectReceiveSlotPool::Slot> slot;
        while (!(slot = pool->try_acquire()).has_value()) {
          std::this_thread::yield();
        }
        slot->reset();
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(completed.load(std::memory_order_relaxed), thread_count * iterations);
  auto const snapshot = pool->snapshot();
  EXPECT_LE(snapshot.allocated_slots, snapshot.max_slots);
  EXPECT_EQ(snapshot.allocating_slots, 0);
  EXPECT_EQ(snapshot.checked_out_slots, 0);
  EXPECT_EQ(snapshot.free_slots, snapshot.allocated_slots);
  EXPECT_EQ(snapshot.quarantined_slots, 0);
  EXPECT_LE(snapshot.allocation_high_water_slots, snapshot.max_slots);
  EXPECT_LE(snapshot.checked_out_high_water_slots, snapshot.max_slots);
  EXPECT_LE(snapshot.reserved_bytes(), snapshot.max_pinned_bytes);
}

TEST_F(DirectReceiveSlotPoolTest, FailedSyncQuarantineIsNeverReusedOrReplacedPastCap)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  auto pool            = DirectReceiveSlotPool::create_for_testing(slot_size, 2 * slot_size);
  auto uncertain       = pool->try_acquire();
  ASSERT_TRUE(uncertain.has_value());
  auto* uncertain_address = uncertain->get();

  pool->quarantine_after_failed_sync(std::move(uncertain.value()));
  EXPECT_FALSE(static_cast<bool>(uncertain.value()));
  auto after_quarantine = pool->snapshot();
  EXPECT_EQ(after_quarantine.allocated_slots, 1);
  EXPECT_EQ(after_quarantine.checked_out_slots, 0);
  EXPECT_EQ(after_quarantine.quarantined_slots, 1);

  auto replacement = pool->try_acquire();
  ASSERT_TRUE(replacement.has_value());
  EXPECT_NE(replacement->get(), uncertain_address);
  EXPECT_FALSE(pool->try_acquire().has_value());

  replacement->reset();
  auto reused_safe_slot = pool->try_acquire();
  ASSERT_TRUE(reused_safe_slot.has_value());
  EXPECT_NE(reused_safe_slot->get(), uncertain_address);
  EXPECT_FALSE(pool->try_acquire().has_value());
}

TEST_F(DirectReceiveSlotPoolTest, SlotCanOutliveAnIsolatedPool)
{
  auto const slot_size = DirectReceiveSlotPool::minimum_slot_size();
  std::optional<DirectReceiveSlotPool::Slot> survivor;
  {
    auto pool = DirectReceiveSlotPool::create_for_testing(slot_size, slot_size);
    survivor  = pool->try_acquire();
    ASSERT_TRUE(survivor.has_value());
    EXPECT_TRUE(static_cast<bool>(survivor.value()));
  }

  // Slot retains the shared state needed to recycle and eventually deallocate its storage.
  survivor->reset();
  EXPECT_FALSE(static_cast<bool>(survivor.value()));
}

TEST_F(DirectReceiveSlotPoolTest, ProductionDefaultsFreezeAtFirstPoolUse)
{
  auto const expected_slot_size = kvikio::defaults::remote_direct_receive_slot_size();
  auto const expected_cap       = kvikio::defaults::remote_direct_receive_max_pinned_bytes();

  auto& pool          = DirectReceiveSlotPool::instance();
  auto const snapshot = pool.snapshot();
  EXPECT_EQ(snapshot.slot_size, expected_slot_size);
  EXPECT_EQ(snapshot.max_pinned_bytes, expected_cap);
  EXPECT_EQ(snapshot.max_slots, expected_cap / expected_slot_size);
  EXPECT_EQ(snapshot.allocated_slots, 0);

  EXPECT_THROW(kvikio::defaults::set_remote_direct_receive_slot_size(expected_slot_size),
               std::logic_error);
  EXPECT_THROW(kvikio::defaults::set_remote_direct_receive_max_pinned_bytes(expected_cap),
               std::logic_error);
}

}  // namespace
