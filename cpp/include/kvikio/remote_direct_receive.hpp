/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include <kvikio/shim/utils.hpp>

namespace kvikio {

/**
 * @brief Counters for the experimental caller-owned remote receive path.
 *
 * Counters are cumulative within one loaded KvikIO library instance and deliberately separate
 * strict RX-kTLS work from copied fallback work. Performance tests can therefore detect a silent
 * change of data path. `reset_remote_direct_receive_stats()` starts a new measurement interval.
 * A transfer is one internal remote subrange, not necessarily one user-facing `pread()`. Raw bytes
 * include HTTP framing consumed from caller-owned receive loans; body bytes include only accepted
 * range payload. Slot exhaustion counts failed acquisition attempts and may repeat while waiting.
 */
struct RemoteDirectReceiveStats {
  std::uint64_t transfers_requested{};
  // Cumulative activations, not a gauge of transfers currently in flight.
  std::uint64_t strict_rx_transfers_activated{};
  std::uint64_t strict_rx_transfers_completed{};
  std::uint64_t transfers_fallback{};
  std::uint64_t transfers_failed{};
  std::uint64_t retries{};
  std::uint64_t cancellations{};
  std::uint64_t pinned_slots_acquired{};
  std::uint64_t pinned_slot_exhaustions{};

  // Receive totals include only successful final attempts. H2D totals include successful
  // submission calls whose containing transfer may subsequently retry or fail.
  std::uint64_t strict_rx_raw_received_bytes{};
  std::uint64_t strict_rx_body_bytes{};
  std::uint64_t strict_rx_h2d_bytes{};
  std::uint64_t strict_rx_h2d_batches{};
  std::uint64_t copied_stream_transfers_completed{};
  std::uint64_t copied_stream_raw_received_bytes{};
  std::uint64_t copied_stream_body_bytes{};
  std::uint64_t copied_stream_h2d_bytes{};
  std::uint64_t copied_stream_h2d_batches{};

  std::uint64_t protocol_validation_failures{};
  std::uint64_t fallback_capability_unavailable{};
  std::uint64_t fallback_ineligible_request{};
};

/**
 * @brief Take a coherent-enough snapshot of direct-receive counters.
 *
 * Individual fields are atomic, but the returned structure is not a transactional snapshot across
 * concurrent transfers. It is intended for before/after deltas around a benchmark.
 */
[[nodiscard]] KVIKIO_EXPORT RemoteDirectReceiveStats remote_direct_receive_stats() noexcept;

/**
 * @brief Reset all direct-receive counters.
 *
 * Call only while no remote requests are active when exact benchmark deltas are required.
 */
KVIKIO_EXPORT void reset_remote_direct_receive_stats() noexcept;

}  // namespace kvikio
