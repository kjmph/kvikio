/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace kvikio::detail {

enum class DirectReceiveFallbackReason : std::uint8_t {
  capability_unavailable,
  ineligible_request,
};

enum class DirectReceiveFailureReason : std::uint8_t {
  other,
  protocol_validation,
};

/** Internal metric increments shared by remote implementations and the public stats API. */
void direct_receive_record_requested() noexcept;
void direct_receive_record_strict_activated() noexcept;
void direct_receive_record_strict_completion(std::size_t raw_bytes,
                                             std::size_t body_bytes) noexcept;
void direct_receive_record_strict_h2d_submission(std::size_t bytes,
                                                 std::size_t batches = 1) noexcept;
void direct_receive_record_copied_completion(std::size_t raw_bytes,
                                             std::size_t body_bytes) noexcept;
void direct_receive_record_copied_h2d_submission(std::size_t bytes,
                                                 std::size_t batches = 1) noexcept;
void direct_receive_record_fallback(DirectReceiveFallbackReason reason) noexcept;
void direct_receive_record_failed(DirectReceiveFailureReason reason) noexcept;
void direct_receive_record_retry() noexcept;
void direct_receive_record_cancellation() noexcept;
void direct_receive_record_slot_acquired() noexcept;
void direct_receive_record_slot_exhaustion() noexcept;

}  // namespace kvikio::detail
