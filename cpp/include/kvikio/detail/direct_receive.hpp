/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <curl/curl.h>

namespace kvikio {
class CurlHandle;
}

namespace kvikio::detail {

/**
 * @brief Whether response body bytes may be accepted as the requested payload.
 *
 * Streaming direct receive must make this decision from a complete HTTP header block before it
 * records any body spans for H2D submission. HTTP/1.1 informational and redirect bodies are
 * consumed but discarded; an exact HTTP/1.1 206 response is accepted only after all Range metadata
 * has been validated.
 */
enum class DirectReceiveBodyDisposition : std::uint8_t {
  undecided,
  discard,
  accept_range,
  reject,
};

struct DirectReceiveSpan {
  std::size_t source_offset{};
  std::size_t destination_offset{};
  std::size_t size{};
};

// A receive callback must never allocate. Bound each slot's retained scatter description so an
// adversarial callback sequence cannot grow memory.
inline constexpr std::size_t direct_receive_max_spans_per_slot = 256;

/**
 * @brief Fixed-capacity description of one released receive slot ready for H2D submission.
 *
 * The descriptor is fixed-size so taking a slot from the curl callback state cannot allocate. Its
 * destination offsets are relative to the complete requested range, not merely this slot.
 */
struct DirectReceiveReleasedSlot {
  std::array<DirectReceiveSpan, direct_receive_max_spans_per_slot> spans{};
  std::size_t span_count{};
  std::size_t raw_bytes{};
  std::size_t body_bytes{};
};

/**
 * @brief Tracks body spans that libcurl exposes from a caller-owned raw receive buffer.
 *
 * HTTP headers and body may share a receive buffer. The write callback therefore cannot assume
 * that the body starts at offset zero; it records source offsets while destination offsets remain
 * densely packed. Adjacent spans are coalesced without copying.
 */
class DirectReceiveSpanTracker {
 public:
  void set_buffer(void* buffer, std::size_t capacity);
  [[nodiscard]] bool record_body(void const* data, std::size_t size) noexcept;
  [[nodiscard]] bool advance_raw(std::size_t raw_bytes) noexcept;
  void reset() noexcept;

  [[nodiscard]] std::span<DirectReceiveSpan const> spans() const noexcept;
  [[nodiscard]] std::size_t body_bytes() const noexcept;
  [[nodiscard]] std::size_t raw_bytes() const noexcept;
  [[nodiscard]] bool contains(void const* data, std::size_t size) const noexcept;
  [[nodiscard]] bool span_capacity_exhausted() const noexcept;

 private:
  std::byte* _buffer{};
  std::size_t _capacity{};
  std::size_t _body_bytes{};
  std::size_t _raw_bytes{};
  std::size_t _span_count{};
  bool _span_capacity_exhausted{};
  std::array<DirectReceiveSpan, direct_receive_max_spans_per_slot> _spans{};
};

/**
 * @brief Per-attempt HTTP response validation for direct receive.
 */
class DirectReceiveResponse {
 public:
  void consume_header(std::string_view line) noexcept;
  void reset() noexcept;
  [[nodiscard]] bool transfer_encoding_seen() const noexcept;
  [[nodiscard]] DirectReceiveBodyDisposition body_disposition(
    std::size_t requested_offset, std::size_t requested_size) const noexcept;
  [[nodiscard]] std::optional<std::string> validate(long response_code,
                                                    long http_version,
                                                    curl_off_t content_length,
                                                    std::size_t requested_offset,
                                                    std::size_t requested_size) const;

 private:
  std::optional<std::size_t> _content_length;
  std::optional<std::size_t> _content_range_start;
  std::optional<std::size_t> _content_range_end;
  std::optional<std::size_t> _content_range_total;
  std::optional<unsigned int> _response_code;
  bool _content_range_seen{};
  bool _content_encoding_identity{true};
  bool _transfer_encoding_seen{};
  bool _http11{};
  bool _header_block_complete{};
  bool _malformed{};
};

/**
 * @brief Whether a failed strict direct-RX attempt may safely retry through ordinary libcurl.
 *
 * Only the patched backend's pre-handoff capability result is eligible. Security, integrity,
 * authentication, transient I/O, and any failure after direct RX became active must remain fatal
 * (or follow the ordinary retry policy) rather than silently changing the measured path.
 */
[[nodiscard]] bool direct_receive_can_fallback(bool required,
                                               CURLcode result,
                                               long direct_status,
                                               std::size_t body_bytes) noexcept;

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)

/**
 * @brief One-transfer bridge between patched libcurl and KvikIO's pinned buffer.
 *
 * A transfer loans consecutive unused regions of one bounded pinned allocation. A release advances
 * the allocation cursor by the raw bytes consumed, so later receive cycles cannot overwrite spans
 * retained for the eventual H2D batch. The accelerated path admits only ranges with enough spare
 * capacity for bounded HTTP headers and never allocates or copies in a callback.
 */
class CurlDirectReceiveState {
 public:
  CurlDirectReceiveState(std::size_t requested_offset, std::size_t requested_size);
  CurlDirectReceiveState(CurlDirectReceiveState const&)            = delete;
  CurlDirectReceiveState& operator=(CurlDirectReceiveState const&) = delete;
  CurlDirectReceiveState(CurlDirectReceiveState&&)                 = delete;
  CurlDirectReceiveState& operator=(CurlDirectReceiveState&&)      = delete;

  /**
   * @brief Configure libcurl to stream one validated Range response through caller-owned slots.
   *
   * @param curl Handle to configure.
   * @param require_ktls When true, require the patched strict Linux RX-kTLS transport. When false,
   * use the same bounded caller-owned slot pipeline with libcurl's ordinary TLS receive path. The
   * latter is the explicit copied fallback for `PREFER`; it never masquerades as direct RX.
   */
  void configure(CurlHandle& curl, bool require_ktls);
  void install_buffer(void* pinned_buffer, std::size_t capacity);
  [[nodiscard]] bool slot_ready() const noexcept;
  [[nodiscard]] bool needs_buffer() const noexcept;
  [[nodiscard]] bool body_complete() const noexcept;
  void finalize_current_slot() noexcept;
  [[nodiscard]] DirectReceiveReleasedSlot take_released_slot();
  [[nodiscard]] std::optional<std::string> validate(CurlHandle& curl) const;

  [[nodiscard]] std::size_t raw_bytes() const noexcept;
  [[nodiscard]] std::size_t body_bytes() const noexcept;
  [[nodiscard]] bool callback_failed() const noexcept;
  [[nodiscard]] bool callback_protocol_validation_failed() const noexcept;
  [[nodiscard]] std::string_view callback_error() const noexcept;

  // Public to permit deterministic lifecycle tests without a socket. libcurl reaches these through
  // the static C callbacks below.
  [[nodiscard]] curl_recv_buffer_result acquire_buffer(std::size_t suggested_size,
                                                       curl_recv_buffer* buffer) noexcept;
  void release_buffer(curl_recv_buffer const* buffer, std::size_t used) noexcept;
  [[nodiscard]] std::size_t consume_body(char* data, std::size_t size, std::size_t nmemb) noexcept;
  [[nodiscard]] std::size_t consume_header(char* data,
                                           std::size_t size,
                                           std::size_t nmemb) noexcept;

 private:
  static curl_recv_buffer_result acquire(CURL* easy,
                                         std::size_t suggested_size,
                                         curl_recv_buffer* buffer,
                                         void* userdata) noexcept;
  static void release(CURL* easy,
                      curl_recv_buffer const* buffer,
                      std::size_t used,
                      void* userdata) noexcept;
  static std::size_t write_body(char* data,
                                std::size_t size,
                                std::size_t nmemb,
                                void* userdata) noexcept;
  static std::size_t write_header(char* data,
                                  std::size_t size,
                                  std::size_t nmemb,
                                  void* userdata) noexcept;

  enum class CallbackError : std::uint8_t {
    none,
    size_overflow,
    invalid_callback_data,
    invalid_acquire,
    invalid_release,
    body_outside_loan,
    body_length_exceeded,
    span_capacity_exhausted,
    transfer_encoding,
    response_not_accepted,
  };

  void fail_callback(CallbackError error) noexcept;
  [[nodiscard]] static std::string_view callback_error_message(CallbackError error) noexcept;

  void* _pinned_buffer{};
  std::size_t _capacity{};
  std::size_t _requested_offset;
  std::size_t _requested_size;
  std::size_t _completed_raw_bytes{};
  std::size_t _completed_body_bytes{};
  DirectReceiveSpanTracker _tracker;
  DirectReceiveResponse _response;
  bool _loan_outstanding{};
  bool _loan_ever_released{};
  void* _loan_buffer{};
  std::size_t _loan_capacity{};
  std::size_t _loan_body_high_water{};
  bool _slot_ready{};
  bool _callback_failed{};
  CallbackError _callback_error{CallbackError::none};
};

#endif

}  // namespace kvikio::detail
