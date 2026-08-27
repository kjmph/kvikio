/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/direct_receive_stats.hpp>
#include <kvikio/remote_direct_receive.hpp>

namespace {

void expect_all_zero(kvikio::RemoteDirectReceiveStats const& stats)
{
  EXPECT_EQ(stats.transfers_requested, 0);
  EXPECT_EQ(stats.strict_rx_transfers_activated, 0);
  EXPECT_EQ(stats.strict_rx_transfers_completed, 0);
  EXPECT_EQ(stats.transfers_fallback, 0);
  EXPECT_EQ(stats.transfers_failed, 0);
  EXPECT_EQ(stats.retries, 0);
  EXPECT_EQ(stats.cancellations, 0);
  EXPECT_EQ(stats.pinned_slots_acquired, 0);
  EXPECT_EQ(stats.pinned_slot_exhaustions, 0);
  EXPECT_EQ(stats.strict_rx_raw_received_bytes, 0);
  EXPECT_EQ(stats.strict_rx_body_bytes, 0);
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 0);
  EXPECT_EQ(stats.strict_rx_h2d_batches, 0);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 0);
  EXPECT_EQ(stats.copied_stream_raw_received_bytes, 0);
  EXPECT_EQ(stats.copied_stream_body_bytes, 0);
  EXPECT_EQ(stats.copied_stream_h2d_bytes, 0);
  EXPECT_EQ(stats.copied_stream_h2d_batches, 0);
  EXPECT_EQ(stats.protocol_validation_failures, 0);
  EXPECT_EQ(stats.fallback_capability_unavailable, 0);
  EXPECT_EQ(stats.fallback_ineligible_request, 0);
}

void expect_all_zero(kvikio::RemoteDirectReceiveHostStats const& stats)
{
  EXPECT_EQ(stats.strict_rx_direct_placement_bytes, 0);
  EXPECT_EQ(stats.strict_rx_framing_compaction_bytes, 0);
  EXPECT_EQ(stats.copied_stream_direct_placement_bytes, 0);
  EXPECT_EQ(stats.copied_stream_framing_compaction_bytes, 0);
}

}  // namespace

TEST(RemoteDirectReceiveStats, distinguishes_strict_and_copied_work)
{
  using kvikio::detail::DirectReceiveFailureReason;
  using kvikio::detail::DirectReceiveFallbackReason;

  kvikio::reset_remote_direct_receive_stats();
  expect_all_zero(kvikio::remote_direct_receive_stats());
  expect_all_zero(kvikio::remote_direct_receive_host_stats());

  kvikio::detail::direct_receive_record_requested();
  kvikio::detail::direct_receive_record_requested();
  kvikio::detail::direct_receive_record_strict_activated();
  kvikio::detail::direct_receive_record_strict_activated();
  kvikio::detail::direct_receive_record_strict_completion(101, 97);
  kvikio::detail::direct_receive_record_strict_completion(211, 199);
  kvikio::detail::direct_receive_record_strict_h2d_submission(257);
  kvikio::detail::direct_receive_record_strict_h2d_submission(509, 3);
  kvikio::detail::direct_receive_record_copied_completion(307, 293);
  kvikio::detail::direct_receive_record_copied_completion(401, 389);
  kvikio::detail::direct_receive_record_copied_h2d_submission(601);
  kvikio::detail::direct_receive_record_copied_h2d_submission(701, 5);
  kvikio::detail::direct_receive_record_strict_host_completion(809, 31);
  kvikio::detail::direct_receive_record_copied_host_completion(907, 37);
  kvikio::detail::direct_receive_record_fallback(
    DirectReceiveFallbackReason::capability_unavailable);
  kvikio::detail::direct_receive_record_fallback(DirectReceiveFallbackReason::ineligible_request);
  kvikio::detail::direct_receive_record_failed(DirectReceiveFailureReason::protocol_validation);
  kvikio::detail::direct_receive_record_failed(DirectReceiveFailureReason::other);
  kvikio::detail::direct_receive_record_retry();
  kvikio::detail::direct_receive_record_retry();
  kvikio::detail::direct_receive_record_cancellation();
  kvikio::detail::direct_receive_record_slot_acquired();
  kvikio::detail::direct_receive_record_slot_acquired();
  kvikio::detail::direct_receive_record_slot_acquired();
  kvikio::detail::direct_receive_record_slot_exhaustion();
  kvikio::detail::direct_receive_record_slot_exhaustion();

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 2);
  EXPECT_EQ(stats.strict_rx_transfers_activated, 2);
  EXPECT_EQ(stats.strict_rx_transfers_completed, 2);
  EXPECT_EQ(stats.transfers_fallback, 2);
  EXPECT_EQ(stats.transfers_failed, 2);
  EXPECT_EQ(stats.retries, 2);
  EXPECT_EQ(stats.cancellations, 1);
  EXPECT_EQ(stats.pinned_slots_acquired, 3);
  EXPECT_EQ(stats.pinned_slot_exhaustions, 2);
  EXPECT_EQ(stats.strict_rx_raw_received_bytes, 312);
  EXPECT_EQ(stats.strict_rx_body_bytes, 296);
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 766);
  EXPECT_EQ(stats.strict_rx_h2d_batches, 4);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 2);
  EXPECT_EQ(stats.copied_stream_raw_received_bytes, 708);
  EXPECT_EQ(stats.copied_stream_body_bytes, 682);
  EXPECT_EQ(stats.copied_stream_h2d_bytes, 1302);
  EXPECT_EQ(stats.copied_stream_h2d_batches, 6);
  EXPECT_EQ(stats.protocol_validation_failures, 1);
  EXPECT_EQ(stats.fallback_capability_unavailable, 1);
  EXPECT_EQ(stats.fallback_ineligible_request, 1);

  auto const host_stats = kvikio::remote_direct_receive_host_stats();
  EXPECT_EQ(host_stats.strict_rx_direct_placement_bytes, 809);
  EXPECT_EQ(host_stats.strict_rx_framing_compaction_bytes, 31);
  EXPECT_EQ(host_stats.copied_stream_direct_placement_bytes, 907);
  EXPECT_EQ(host_stats.copied_stream_framing_compaction_bytes, 37);

  kvikio::reset_remote_direct_receive_stats();
  expect_all_zero(kvikio::remote_direct_receive_stats());
  expect_all_zero(kvikio::remote_direct_receive_host_stats());
}

TEST(RemoteDirectReceiveStats, concurrent_recorders_do_not_lose_updates)
{
  constexpr std::uint64_t num_threads = 8;
  constexpr std::uint64_t iterations  = 2000;
  kvikio::reset_remote_direct_receive_stats();

  {
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (std::uint64_t thread = 0; thread < num_threads; ++thread) {
      threads.emplace_back([] {
        for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
          kvikio::detail::direct_receive_record_requested();
          kvikio::detail::direct_receive_record_strict_activated();
          kvikio::detail::direct_receive_record_strict_completion(3, 2);
          kvikio::detail::direct_receive_record_strict_h2d_submission(2);
          kvikio::detail::direct_receive_record_strict_host_completion(5, 1);
          kvikio::detail::direct_receive_record_slot_acquired();
        }
      });
    }
  }

  auto const expected = num_threads * iterations;
  auto const stats    = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, expected);
  EXPECT_EQ(stats.strict_rx_transfers_activated, expected);
  EXPECT_EQ(stats.strict_rx_transfers_completed, expected);
  EXPECT_EQ(stats.strict_rx_raw_received_bytes, 3 * expected);
  EXPECT_EQ(stats.strict_rx_body_bytes, 2 * expected);
  EXPECT_EQ(stats.strict_rx_h2d_bytes, 2 * expected);
  EXPECT_EQ(stats.strict_rx_h2d_batches, expected);
  EXPECT_EQ(stats.pinned_slots_acquired, expected);
  auto const host_stats = kvikio::remote_direct_receive_host_stats();
  EXPECT_EQ(host_stats.strict_rx_direct_placement_bytes, 5 * expected);
  EXPECT_EQ(host_stats.strict_rx_framing_compaction_bytes, expected);
}
