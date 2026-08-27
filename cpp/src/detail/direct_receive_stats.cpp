/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstdint>

#include <kvikio/detail/direct_receive_stats.hpp>
#include <kvikio/remote_direct_receive.hpp>

namespace kvikio {
namespace {

struct AtomicDirectReceiveStats {
  std::atomic<std::uint64_t> transfers_requested{};
  std::atomic<std::uint64_t> strict_rx_transfers_activated{};
  std::atomic<std::uint64_t> strict_rx_transfers_completed{};
  std::atomic<std::uint64_t> transfers_fallback{};
  std::atomic<std::uint64_t> transfers_failed{};
  std::atomic<std::uint64_t> retries{};
  std::atomic<std::uint64_t> cancellations{};
  std::atomic<std::uint64_t> pinned_slots_acquired{};
  std::atomic<std::uint64_t> pinned_slot_exhaustions{};
  std::atomic<std::uint64_t> strict_rx_raw_received_bytes{};
  std::atomic<std::uint64_t> strict_rx_body_bytes{};
  std::atomic<std::uint64_t> strict_rx_h2d_bytes{};
  std::atomic<std::uint64_t> strict_rx_h2d_batches{};
  std::atomic<std::uint64_t> copied_stream_transfers_completed{};
  std::atomic<std::uint64_t> copied_stream_raw_received_bytes{};
  std::atomic<std::uint64_t> copied_stream_body_bytes{};
  std::atomic<std::uint64_t> copied_stream_h2d_bytes{};
  std::atomic<std::uint64_t> copied_stream_h2d_batches{};
  std::atomic<std::uint64_t> protocol_validation_failures{};
  std::atomic<std::uint64_t> fallback_capability_unavailable{};
  std::atomic<std::uint64_t> fallback_ineligible_request{};
};

AtomicDirectReceiveStats& atomic_stats() noexcept
{
  static AtomicDirectReceiveStats stats;
  return stats;
}

std::uint64_t load(std::atomic<std::uint64_t> const& value) noexcept
{
  return value.load(std::memory_order_relaxed);
}

void clear(std::atomic<std::uint64_t>& value) noexcept
{
  value.store(0, std::memory_order_relaxed);
}

void increment(std::atomic<std::uint64_t>& value, std::uint64_t amount = 1) noexcept
{
  value.fetch_add(amount, std::memory_order_relaxed);
}

}  // namespace

RemoteDirectReceiveStats remote_direct_receive_stats() noexcept
{
  auto const& s = atomic_stats();
  return {.transfers_requested               = load(s.transfers_requested),
          .strict_rx_transfers_activated     = load(s.strict_rx_transfers_activated),
          .strict_rx_transfers_completed     = load(s.strict_rx_transfers_completed),
          .transfers_fallback                = load(s.transfers_fallback),
          .transfers_failed                  = load(s.transfers_failed),
          .retries                           = load(s.retries),
          .cancellations                     = load(s.cancellations),
          .pinned_slots_acquired             = load(s.pinned_slots_acquired),
          .pinned_slot_exhaustions           = load(s.pinned_slot_exhaustions),
          .strict_rx_raw_received_bytes      = load(s.strict_rx_raw_received_bytes),
          .strict_rx_body_bytes              = load(s.strict_rx_body_bytes),
          .strict_rx_h2d_bytes               = load(s.strict_rx_h2d_bytes),
          .strict_rx_h2d_batches             = load(s.strict_rx_h2d_batches),
          .copied_stream_transfers_completed = load(s.copied_stream_transfers_completed),
          .copied_stream_raw_received_bytes  = load(s.copied_stream_raw_received_bytes),
          .copied_stream_body_bytes          = load(s.copied_stream_body_bytes),
          .copied_stream_h2d_bytes           = load(s.copied_stream_h2d_bytes),
          .copied_stream_h2d_batches         = load(s.copied_stream_h2d_batches),
          .protocol_validation_failures      = load(s.protocol_validation_failures),
          .fallback_capability_unavailable   = load(s.fallback_capability_unavailable),
          .fallback_ineligible_request       = load(s.fallback_ineligible_request)};
}

void reset_remote_direct_receive_stats() noexcept
{
  auto& s = atomic_stats();
  clear(s.transfers_requested);
  clear(s.strict_rx_transfers_activated);
  clear(s.strict_rx_transfers_completed);
  clear(s.transfers_fallback);
  clear(s.transfers_failed);
  clear(s.retries);
  clear(s.cancellations);
  clear(s.pinned_slots_acquired);
  clear(s.pinned_slot_exhaustions);
  clear(s.strict_rx_raw_received_bytes);
  clear(s.strict_rx_body_bytes);
  clear(s.strict_rx_h2d_bytes);
  clear(s.strict_rx_h2d_batches);
  clear(s.copied_stream_transfers_completed);
  clear(s.copied_stream_raw_received_bytes);
  clear(s.copied_stream_body_bytes);
  clear(s.copied_stream_h2d_bytes);
  clear(s.copied_stream_h2d_batches);
  clear(s.protocol_validation_failures);
  clear(s.fallback_capability_unavailable);
  clear(s.fallback_ineligible_request);
}

namespace detail {

void direct_receive_record_requested() noexcept { increment(atomic_stats().transfers_requested); }
void direct_receive_record_strict_activated() noexcept
{
  increment(atomic_stats().strict_rx_transfers_activated);
}
void direct_receive_record_strict_completion(std::size_t raw_bytes, std::size_t body_bytes) noexcept
{
  auto& s = atomic_stats();
  increment(s.strict_rx_transfers_completed);
  increment(s.strict_rx_raw_received_bytes, raw_bytes);
  increment(s.strict_rx_body_bytes, body_bytes);
}
void direct_receive_record_strict_h2d_submission(std::size_t bytes, std::size_t batches) noexcept
{
  auto& s = atomic_stats();
  increment(s.strict_rx_h2d_bytes, bytes);
  increment(s.strict_rx_h2d_batches, batches);
}
void direct_receive_record_copied_completion(std::size_t raw_bytes, std::size_t body_bytes) noexcept
{
  auto& s = atomic_stats();
  increment(s.copied_stream_transfers_completed);
  increment(s.copied_stream_raw_received_bytes, raw_bytes);
  increment(s.copied_stream_body_bytes, body_bytes);
}
void direct_receive_record_copied_h2d_submission(std::size_t bytes, std::size_t batches) noexcept
{
  auto& s = atomic_stats();
  increment(s.copied_stream_h2d_bytes, bytes);
  increment(s.copied_stream_h2d_batches, batches);
}
void direct_receive_record_fallback(DirectReceiveFallbackReason reason) noexcept
{
  auto& s = atomic_stats();
  increment(s.transfers_fallback);
  increment(reason == DirectReceiveFallbackReason::capability_unavailable
              ? s.fallback_capability_unavailable
              : s.fallback_ineligible_request);
}
void direct_receive_record_failed(DirectReceiveFailureReason reason) noexcept
{
  auto& s = atomic_stats();
  increment(s.transfers_failed);
  if (reason == DirectReceiveFailureReason::protocol_validation) {
    increment(s.protocol_validation_failures);
  }
}
void direct_receive_record_retry() noexcept { increment(atomic_stats().retries); }
void direct_receive_record_cancellation() noexcept { increment(atomic_stats().cancellations); }
void direct_receive_record_slot_acquired() noexcept
{
  increment(atomic_stats().pinned_slots_acquired);
}
void direct_receive_record_slot_exhaustion() noexcept
{
  increment(atomic_stats().pinned_slot_exhaustions);
}

}  // namespace detail
}  // namespace kvikio
