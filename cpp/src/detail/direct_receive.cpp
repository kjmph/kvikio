/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <curl/curl.h>

#include <kvikio/detail/direct_receive.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

#if !defined(CURL_HAS_RECV_BUFFER_CALLBACKS) || !defined(CURL_HAS_KTLS_DIRECT_RX)
#error "KvikIO remote direct receive requires caller-owned buffers and strict RX kTLS in libcurl"
#endif

namespace kvikio {
namespace detail {
namespace {

bool is_ows(char value) noexcept { return value == ' ' || value == '\t'; }

bool is_ascii_alpha(char value) noexcept
{
  return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

bool is_ascii_digit(char value) noexcept { return value >= '0' && value <= '9'; }

char ascii_lower(char value) noexcept
{
  return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A')) : value;
}

std::string_view trim_ows(std::string_view value) noexcept
{
  while (!value.empty() && is_ows(value.front())) {
    value.remove_prefix(1);
  }
  while (!value.empty() && is_ows(value.back())) {
    value.remove_suffix(1);
  }
  return value;
}

bool is_field_name_character(char value) noexcept
{
  if (is_ascii_alpha(value) || is_ascii_digit(value)) { return true; }
  switch (value) {
    case '!':
    case '#':
    case '$':
    case '%':
    case '&':
    case '\'':
    case '*':
    case '+':
    case '-':
    case '.':
    case '^':
    case '_':
    case '`':
    case '|':
    case '~': return true;
    default: return false;
  }
}

bool iequals(std::string_view lhs, std::string_view rhs) noexcept
{
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](char a, char b) {
           return ascii_lower(a) == ascii_lower(b);
         });
}

std::optional<std::size_t> parse_size(std::string_view value) noexcept
{
  value = trim_ows(value);
  std::size_t parsed{};
  auto const [end, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{} || end != value.data() + value.size()) { return std::nullopt; }
  return parsed;
}

bool is_strong_entity_tag(std::string_view value) noexcept
{
  if (value.size() < 2 || value.size() > direct_receive_max_entity_tag_size ||
      value.front() != '"' || value.back() != '"') {
    return false;
  }
  // RFC 9110 entity-tag = [ weak ] opaque-tag. If-Match uses the strong
  // comparison function, so accepting W/ here would make every conditional
  // retry fail and would not prove object identity.
  for (auto const value_byte : value.substr(1, value.size() - 2)) {
    auto const byte = static_cast<unsigned char>(value_byte);
    if (byte == 0x21 || (byte >= 0x23 && byte <= 0x7e) || byte >= 0x80) { continue; }
    return false;
  }
  return true;
}

}  // namespace

DirectReceiveObjectSnapshot::DirectReceiveObjectSnapshot(std::size_t expected_size,
                                                         bool require_entity_tag,
                                                         bool send_if_match) noexcept
  : _expected_size{expected_size},
    _require_entity_tag{require_entity_tag},
    _send_if_match{send_if_match}
{
}

std::size_t DirectReceiveObjectSnapshot::expected_size() const noexcept { return _expected_size; }

bool DirectReceiveObjectSnapshot::requires_entity_tag() const noexcept
{
  return _require_entity_tag;
}

bool DirectReceiveObjectSnapshot::accept_entity_tag(
  std::optional<std::string_view> entity_tag) noexcept
{
  if (!_require_entity_tag) { return true; }
  if (!entity_tag.has_value() || !is_strong_entity_tag(entity_tag.value())) { return false; }
  try {
    std::lock_guard const lock{_mutex};
    if (_entity_tag_size == 0) {
      std::copy(entity_tag->begin(), entity_tag->end(), _entity_tag.begin());
      _entity_tag_size = entity_tag->size();
      return true;
    }
    return entity_tag->size() == _entity_tag_size &&
           std::equal(entity_tag->begin(), entity_tag->end(), _entity_tag.begin());
  } catch (...) {
    return false;
  }
}

std::optional<std::string> DirectReceiveObjectSnapshot::if_match_entity_tag() const
{
  std::lock_guard const lock{_mutex};
  if (!_send_if_match || _entity_tag_size == 0) { return std::nullopt; }
  return std::string{_entity_tag.data(), _entity_tag_size};
}

DirectReceiveHostPlacement place_direct_receive_on_host(void const* source,
                                                        std::size_t source_capacity,
                                                        DirectReceiveReleasedSlot const& released,
                                                        void* destination,
                                                        std::size_t destination_extent,
                                                        std::size_t expected_destination_offset,
                                                        bool direct_destination_buffer)
{
  KVIKIO_EXPECT(source != nullptr, "direct-receive host source is null", std::logic_error);
  KVIKIO_EXPECT(
    destination != nullptr, "direct-receive host destination is null", std::logic_error);
  KVIKIO_EXPECT(released.span_count <= released.spans.size(),
                "direct-receive host span count exceeds its descriptor capacity",
                std::logic_error);
  KVIKIO_EXPECT(released.raw_bytes <= source_capacity,
                "direct-receive host source exceeds its receive buffer",
                std::logic_error);
  KVIKIO_EXPECT(expected_destination_offset <= destination_extent,
                "direct-receive host body cursor exceeds its destination",
                std::logic_error);
  if (direct_destination_buffer) {
    KVIKIO_EXPECT(source == static_cast<std::byte*>(destination) + expected_destination_offset,
                  "direct-receive host buffer does not begin at its final destination",
                  std::logic_error);
    KVIKIO_EXPECT(released.raw_bytes == released.body_bytes,
                  "direct-receive host destination contains non-body bytes",
                  std::logic_error);
  }

  std::size_t body_bytes{};
  std::size_t previous_source_end{};
  std::size_t previous_destination_end = expected_destination_offset;
  for (std::size_t i = 0; i < released.span_count; ++i) {
    auto const& span = released.spans[i];
    KVIKIO_EXPECT(span.size != 0,
                  "direct-receive host placement cannot contain an empty span",
                  std::logic_error);
    KVIKIO_EXPECT(span.source_offset <= released.raw_bytes &&
                    span.size <= released.raw_bytes - span.source_offset,
                  "direct-receive host source span exceeds received bytes",
                  std::logic_error);
    KVIKIO_EXPECT(span.destination_offset == previous_destination_end &&
                    span.destination_offset <= destination_extent &&
                    span.size <= destination_extent - span.destination_offset,
                  "direct-receive host destination spans are not contiguous and in bounds",
                  std::logic_error);
    KVIKIO_EXPECT(i == 0 || span.source_offset >= previous_source_end,
                  "direct-receive host source spans overlap or are out of order",
                  std::logic_error);
    KVIKIO_EXPECT(body_bytes <= std::numeric_limits<std::size_t>::max() - span.size,
                  "direct-receive host body byte count overflow",
                  std::logic_error);

    if (direct_destination_buffer) {
      KVIKIO_EXPECT(span.source_offset == body_bytes,
                    "direct-receive host body is not a contiguous raw prefix",
                    std::logic_error);
    }

    body_bytes += span.size;
    previous_source_end      = span.source_offset + span.size;
    previous_destination_end = span.destination_offset + span.size;
  }
  KVIKIO_EXPECT(body_bytes == released.body_bytes,
                "direct-receive host span sizes do not match body bytes",
                std::logic_error);
  KVIKIO_EXPECT(previous_destination_end == expected_destination_offset + body_bytes,
                "direct-receive host placement did not advance by its body bytes",
                std::logic_error);

  // All descriptors are valid before the first mutation. Copy accepted body spans out of the
  // bounded framing window only after that validation. Direct-tail spans are already at their
  // final offsets.
  if (!direct_destination_buffer) {
    for (std::size_t i = 0; i < released.span_count; ++i) {
      auto const& span = released.spans[i];
      std::memmove(static_cast<std::byte*>(destination) + span.destination_offset,
                   static_cast<std::byte const*>(source) + span.source_offset,
                   span.size);
    }
  }

  return direct_destination_buffer ? DirectReceiveHostPlacement{.direct_bytes = body_bytes}
                                   : DirectReceiveHostPlacement{.staged_bytes = body_bytes};
}

void DirectReceiveSpanTracker::set_buffer(void* buffer, std::size_t capacity)
{
  _buffer   = static_cast<std::byte*>(buffer);
  _capacity = capacity;
  reset();
}

bool DirectReceiveSpanTracker::contains(void const* data, std::size_t size) const noexcept
{
  if (_buffer == nullptr || data == nullptr) { return false; }
  auto const base = reinterpret_cast<std::uintptr_t>(_buffer);
  auto const ptr  = reinterpret_cast<std::uintptr_t>(data);
  if (ptr < base || ptr - base > _capacity) { return false; }
  return size <= _capacity - (ptr - base);
}

bool DirectReceiveSpanTracker::record_body(void const* data, std::size_t size) noexcept
{
  if (size == 0) { return true; }
  if (!contains(data, size)) { return false; }
  auto const source_offset = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(data) -
                                                      reinterpret_cast<std::uintptr_t>(_buffer));
  if (_span_count != 0) {
    auto& last = _spans[_span_count - 1];
    if (last.source_offset + last.size == source_offset &&
        last.destination_offset + last.size == _body_bytes) {
      last.size += size;
      _body_bytes += size;
      return true;
    }
  }
  if (_span_count == _spans.size()) {
    _span_capacity_exhausted = true;
    return false;
  }
  _spans[_span_count++] = {source_offset, _body_bytes, size};
  _body_bytes += size;
  return true;
}

bool DirectReceiveSpanTracker::advance_raw(std::size_t raw_bytes) noexcept
{
  if (_raw_bytes > _capacity || raw_bytes > _capacity - _raw_bytes) { return false; }
  _raw_bytes += raw_bytes;
  return true;
}

void DirectReceiveSpanTracker::reset() noexcept
{
  _body_bytes              = 0;
  _raw_bytes               = 0;
  _span_count              = 0;
  _span_capacity_exhausted = false;
}

std::span<DirectReceiveSpan const> DirectReceiveSpanTracker::spans() const noexcept
{
  return {_spans.data(), _span_count};
}

std::size_t DirectReceiveSpanTracker::body_bytes() const noexcept { return _body_bytes; }
std::size_t DirectReceiveSpanTracker::raw_bytes() const noexcept { return _raw_bytes; }
bool DirectReceiveSpanTracker::span_capacity_exhausted() const noexcept
{
  return _span_capacity_exhausted;
}

void DirectReceiveResponse::consume_header(std::string_view line) noexcept
{
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.remove_suffix(1);
  }
  if (line.empty()) {
    _header_block_complete = true;
    return;
  }
  if (line.size() >= 5 && line.substr(0, 5) == "HTTP/") {
    // A subsequent response (for example informational or authentication replay) starts a fresh
    // block. Do not carry validation state across response boundaries.
    reset();
    auto const first_space = line.find(' ');
    if (first_space == std::string_view::npos) {
      _malformed = true;
      return;
    }
    _http11               = iequals(line.substr(0, first_space), "HTTP/1.1");
    auto status           = trim_ows(line.substr(first_space + 1));
    auto const status_end = status.find(' ');
    if (status_end != std::string_view::npos) { status = status.substr(0, status_end); }
    auto const parsed = parse_size(status);
    if (status.size() != 3 || !parsed.has_value() || parsed.value() < 100 || parsed.value() > 999) {
      _malformed = true;
      return;
    }
    _response_code = static_cast<unsigned int>(parsed.value());
    return;
  }

  if (_header_block_complete || line.front() == ' ' || line.front() == '\t') {
    _malformed = true;
    return;
  }

  auto const colon = line.find(':');
  if (colon == std::string_view::npos) {
    _malformed = true;
    return;
  }
  auto const name = line.substr(0, colon);
  if (name.empty() ||
      !std::all_of(name.begin(), name.end(), [](char c) { return is_field_name_character(c); })) {
    _malformed = true;
    return;
  }
  auto const value = trim_ows(line.substr(colon + 1));

  if (iequals(name, "Content-Length")) {
    auto parsed = parse_size(value);
    if (!parsed.has_value() || (_content_length.has_value() && _content_length != parsed)) {
      _malformed = true;
    } else {
      _content_length = parsed;
    }
    return;
  }
  if (iequals(name, "Content-Encoding")) {
    if (value.empty()) { _malformed = true; }
    _content_encoding_identity = _content_encoding_identity && iequals(value, "identity");
    return;
  }
  if (iequals(name, "Transfer-Encoding")) {
    _transfer_encoding_seen = true;
    if (value.empty()) { _malformed = true; }
    return;
  }
  if (iequals(name, "Content-Range")) {
    if (_content_range_seen) {
      _malformed = true;
      return;
    }
    _content_range_seen = true;
    // Strictly accept: bytes <first>-<last>/<size>
    if (value.size() < 7 || !iequals(value.substr(0, 5), "bytes") || value[5] != ' ') {
      _malformed = true;
      return;
    }
    auto range       = value.substr(6);
    auto const dash  = range.find('-');
    auto const slash = range.find('/');
    if (dash == std::string_view::npos || slash == std::string_view::npos || dash >= slash) {
      _malformed = true;
      return;
    }
    auto const start = range.substr(0, dash);
    auto const end   = range.substr(dash + 1, slash - dash - 1);
    auto const total = range.substr(slash + 1);
    if (start != trim_ows(start) || end != trim_ows(end) || total != trim_ows(total)) {
      _malformed = true;
      return;
    }
    _content_range_start = parse_size(start);
    _content_range_end   = parse_size(end);
    _content_range_total = parse_size(total);
    if (!_content_range_start.has_value() || !_content_range_end.has_value() ||
        !_content_range_total.has_value() ||
        _content_range_start.value_or(1) > _content_range_end.value_or(0) ||
        _content_range_total.value_or(0) <= _content_range_end.value_or(0)) {
      _malformed = true;
    }
    return;
  }
  if (iequals(name, "ETag")) {
    if (_entity_tag_seen) {
      _entity_tag_invalid = true;
      return;
    }
    _entity_tag_seen = true;
    if (!is_strong_entity_tag(value)) {
      _entity_tag_invalid = true;
      return;
    }
    std::copy(value.begin(), value.end(), _entity_tag.begin());
    _entity_tag_size = value.size();
  }
}

void DirectReceiveResponse::reset() noexcept
{
  _content_length.reset();
  _content_range_start.reset();
  _content_range_end.reset();
  _content_range_total.reset();
  _response_code.reset();
  _entity_tag_size           = 0;
  _content_range_seen        = false;
  _entity_tag_seen           = false;
  _entity_tag_invalid        = false;
  _content_encoding_identity = true;
  _transfer_encoding_seen    = false;
  _http11                    = false;
  _header_block_complete     = false;
  _malformed                 = false;
}

bool DirectReceiveResponse::transfer_encoding_seen() const noexcept
{
  return _transfer_encoding_seen;
}

bool DirectReceiveResponse::is_partial_content_response() const noexcept
{
  return _response_code == 206;
}

DirectReceiveBodyDisposition DirectReceiveResponse::body_disposition(
  std::size_t requested_offset,
  std::size_t requested_size,
  std::size_t expected_object_size,
  bool require_entity_tag) const noexcept
{
  if (!_header_block_complete) { return DirectReceiveBodyDisposition::undecided; }
  if (_malformed || !_response_code.has_value() || !_http11) {
    return DirectReceiveBodyDisposition::reject;
  }

  auto const response_code = _response_code.value();
  if ((response_code >= 100 && response_code < 200) ||
      (response_code >= 300 && response_code < 400)) {
    return DirectReceiveBodyDisposition::discard;
  }
  if (!_content_encoding_identity || _transfer_encoding_seen || response_code != 206 ||
      requested_size == 0 || !_content_length.has_value() ||
      _content_length.value() != requested_size || !_content_range_seen ||
      !_content_range_start.has_value() || !_content_range_end.has_value() ||
      !_content_range_total.has_value() || _content_range_start.value() != requested_offset ||
      _content_range_end.value() < _content_range_start.value() ||
      _content_range_end.value() - _content_range_start.value() != requested_size - 1 ||
      _content_range_total.value() != expected_object_size ||
      (require_entity_tag && (!_entity_tag_seen || _entity_tag_invalid || _entity_tag_size == 0))) {
    return DirectReceiveBodyDisposition::reject;
  }
  return DirectReceiveBodyDisposition::accept_range;
}

std::optional<std::string> DirectReceiveResponse::validate(long response_code,
                                                           long http_version,
                                                           curl_off_t content_length,
                                                           std::size_t requested_offset,
                                                           std::size_t requested_size,
                                                           std::size_t expected_object_size,
                                                           bool require_entity_tag) const
{
  if (body_disposition(
        requested_offset, requested_size, expected_object_size, require_entity_tag) !=
      DirectReceiveBodyDisposition::accept_range) {
    return "direct receive rejected the response before body handoff";
  }
  if (response_code != 206) { return "direct receive requires an HTTP 206 Range response"; }
  if (http_version != CURL_HTTP_VERSION_1_1) { return "direct receive requires HTTP/1.1"; }
  if (!_content_encoding_identity) { return "direct receive requires identity content encoding"; }
  if (_transfer_encoding_seen) { return "direct receive does not support Transfer-Encoding"; }
  if (!_content_length.has_value() || _content_length.value() != requested_size ||
      content_length < 0 || static_cast<std::uint64_t>(content_length) != requested_size) {
    return "direct receive Content-Length does not match the requested range";
  }
  if (!_content_range_seen || !_content_range_start.has_value() ||
      !_content_range_end.has_value() || !_content_range_total.has_value() ||
      _content_range_start.value() != requested_offset ||
      _content_range_end.value() < _content_range_start.value() || requested_size == 0 ||
      _content_range_end.value() - _content_range_start.value() != requested_size - 1) {
    return "direct receive Content-Range does not match the requested range";
  }
  if (_content_range_total.value() != expected_object_size) {
    return "direct receive Content-Range does not match the object size snapshot";
  }
  if (require_entity_tag && (!_entity_tag_seen || _entity_tag_invalid || _entity_tag_size == 0)) {
    return "direct receive requires one strong ETag";
  }
  return std::nullopt;
}

std::optional<std::string_view> DirectReceiveResponse::entity_tag() const noexcept
{
  if (!_entity_tag_seen || _entity_tag_invalid || _entity_tag_size == 0) { return std::nullopt; }
  return std::string_view{_entity_tag.data(), _entity_tag_size};
}

bool direct_receive_can_fallback(bool required,
                                 CURLcode result,
                                 long direct_status,
                                 std::size_t body_bytes) noexcept
{
#if defined(CURL_HAS_KTLS_DIRECT_RX)
  return !required && result == CURLE_NOT_BUILT_IN &&
         direct_status == CURL_KTLS_DIRECT_RX_UNAVAILABLE && body_bytes == 0;
#else
  static_cast<void>(required);
  static_cast<void>(result);
  static_cast<void>(direct_status);
  static_cast<void>(body_bytes);
  return false;
#endif
}

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)

CurlDirectReceiveState::CurlDirectReceiveState(
  std::size_t requested_offset,
  std::size_t requested_size,
  std::shared_ptr<DirectReceiveObjectSnapshot> object_snapshot)
  : _requested_offset{requested_offset},
    _requested_size{requested_size},
    _object_snapshot{std::move(object_snapshot)}
{
  if (requested_size == 0 ||
      requested_offset > std::numeric_limits<std::size_t>::max() - (requested_size - 1)) {
    throw std::invalid_argument("direct receive requires a nonempty representable byte range");
  }
  if (_object_snapshot == nullptr || requested_offset > _object_snapshot->expected_size() ||
      requested_size > _object_snapshot->expected_size() - requested_offset) {
    throw std::invalid_argument("direct receive range exceeds its object size snapshot");
  }
  _tracker.set_buffer(nullptr, 0);
}

void CurlDirectReceiveState::configure(CurlHandle& curl, bool require_ktls)
{
  curl.setopt(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
  // Use the dedicated option rather than replacing CURLOPT_SSL_OPTIONS, which could silently
  // clear unrelated TLS policy already configured by the endpoint or application.
  curl.setopt(CURLOPT_SSL_KTLS_DIRECT_RX, require_ktls ? 1L : 0L);
  curl.setopt(CURLOPT_HTTP_CONTENT_DECODING, 0L);
  curl.setopt(CURLOPT_RECVBUFFERFUNCTION, &CurlDirectReceiveState::acquire);
  curl.setopt(CURLOPT_RECVBUFFERRELEASEFUNCTION, &CurlDirectReceiveState::release);
  curl.setopt(CURLOPT_RECVBUFFERDATA, this);
  curl.setopt(CURLOPT_WRITEFUNCTION, &CurlDirectReceiveState::write_body);
  curl.setopt(CURLOPT_WRITEDATA, this);
  curl.setopt(CURLOPT_HEADERFUNCTION, &CurlDirectReceiveState::write_header);
  curl.setopt(CURLOPT_HEADERDATA, this);
}

void CurlDirectReceiveState::install_buffer(void* receive_buffer, std::size_t capacity)
{
  if (capacity < DirectReceiveSlotPool::minimum_slot_size()) {
    throw std::logic_error("cannot install a direct-receive slot in the current state");
  }
  install_buffer(receive_buffer, capacity, true);
}

void CurlDirectReceiveState::install_buffer(void* receive_buffer,
                                            std::size_t capacity,
                                            bool rotate_before_short_loan)
{
  if (receive_buffer == nullptr || capacity == 0 || _receive_buffer != nullptr || _slot_ready ||
      _loan_outstanding || _callback_failed || body_complete()) {
    throw std::logic_error("cannot install a direct-receive slot in the current state");
  }
  _receive_buffer           = receive_buffer;
  _capacity                 = capacity;
  _loan_buffer              = nullptr;
  _loan_capacity            = 0;
  _loan_body_high_water     = 0;
  _rotate_before_short_loan = rotate_before_short_loan;
  _tracker.set_buffer(receive_buffer, capacity);
}

bool CurlDirectReceiveState::slot_ready() const noexcept
{
  return !_callback_failed && _slot_ready;
}
bool CurlDirectReceiveState::needs_buffer() const noexcept
{
  return !_callback_failed && _receive_buffer == nullptr && !_slot_ready &&
         body_bytes() < _requested_size;
}
bool CurlDirectReceiveState::body_complete() const noexcept
{
  return body_bytes() == _requested_size;
}
bool CurlDirectReceiveState::response_body_accepted() noexcept
{
  return accept_response_snapshot();
}

void CurlDirectReceiveState::finalize_current_slot() noexcept
{
  if (_callback_failed || _loan_outstanding) {
    fail_callback(CallbackError::invalid_release);
    return;
  }
  if (_receive_buffer != nullptr && _tracker.raw_bytes() != 0) { _slot_ready = true; }
}

DirectReceiveReleasedSlot CurlDirectReceiveState::take_released_slot()
{
  if (_callback_failed || !_slot_ready || _loan_outstanding || _receive_buffer == nullptr) {
    throw std::logic_error("no released direct-receive slot is ready");
  }

  DirectReceiveReleasedSlot released;
  auto const spans    = _tracker.spans();
  released.span_count = spans.size();
  released.raw_bytes  = _tracker.raw_bytes();
  released.body_bytes = _tracker.body_bytes();
  for (std::size_t i = 0; i < spans.size(); ++i) {
    released.spans[i] = spans[i];
    released.spans[i].destination_offset += _completed_body_bytes;
  }

  _completed_raw_bytes += released.raw_bytes;
  _completed_body_bytes += released.body_bytes;
  _receive_buffer       = nullptr;
  _capacity             = 0;
  _slot_ready           = false;
  _loan_buffer          = nullptr;
  _loan_capacity        = 0;
  _loan_body_high_water = 0;
  _tracker.set_buffer(nullptr, 0);
  return released;
}

std::optional<std::string> CurlDirectReceiveState::validate(CurlHandle& curl) const
{
  if (_callback_failed) { return std::string{callback_error()}; }
  if (_loan_outstanding || !_loan_ever_released) {
    return "libcurl did not complete the caller-owned receive-buffer lifecycle";
  }
  if (_tracker.raw_bytes() > _capacity) { return "libcurl reported receive-buffer overflow"; }
  if (body_bytes() != _requested_size) {
    return "direct-receive body length does not match the requested range";
  }
  if (!_snapshot_validation_attempted || !_snapshot_accepted) {
    return "direct-receive response does not match the object version snapshot";
  }

  long response_code{};
  long http_version{};
  curl_off_t content_length{-1};
  curl.getinfo(CURLINFO_RESPONSE_CODE, &response_code);
  curl.getinfo(CURLINFO_HTTP_VERSION, &http_version);
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
  return _response.validate(response_code,
                            http_version,
                            content_length,
                            _requested_offset,
                            _requested_size,
                            _object_snapshot->expected_size(),
                            _object_snapshot->requires_entity_tag());
}

std::size_t CurlDirectReceiveState::raw_bytes() const noexcept
{
  return _completed_raw_bytes + _tracker.raw_bytes();
}
std::size_t CurlDirectReceiveState::body_bytes() const noexcept
{
  return _completed_body_bytes + _tracker.body_bytes();
}
bool CurlDirectReceiveState::callback_failed() const noexcept { return _callback_failed; }
bool CurlDirectReceiveState::callback_protocol_validation_failed() const noexcept
{
  return _callback_error == CallbackError::body_length_exceeded ||
         _callback_error == CallbackError::transfer_encoding ||
         _callback_error == CallbackError::response_not_accepted ||
         _callback_error == CallbackError::object_snapshot_mismatch;
}
std::string_view CurlDirectReceiveState::callback_error() const noexcept
{
  return callback_error_message(_callback_error);
}

curl_recv_buffer_result CurlDirectReceiveState::acquire(CURL*,
                                                        std::size_t suggested_size,
                                                        curl_recv_buffer* buffer,
                                                        void* userdata) noexcept
{
  auto* self = static_cast<CurlDirectReceiveState*>(userdata);
  if (self == nullptr) { return CURL_RECV_BUFFER_ERROR; }
  return self->acquire_buffer(suggested_size, buffer);
}

void CurlDirectReceiveState::release(CURL*,
                                     curl_recv_buffer const* buffer,
                                     std::size_t used,
                                     void* userdata) noexcept
{
  auto* self = static_cast<CurlDirectReceiveState*>(userdata);
  if (self != nullptr) { self->release_buffer(buffer, used); }
}

curl_recv_buffer_result CurlDirectReceiveState::acquire_buffer(std::size_t suggested_size,
                                                               curl_recv_buffer* buffer) noexcept
{
  if (_callback_failed) { return CURL_RECV_BUFFER_ERROR; }
  if (buffer == nullptr || _loan_outstanding || _tracker.raw_bytes() > _capacity) {
    fail_callback(CallbackError::invalid_acquire);
    return CURL_RECV_BUFFER_ERROR;
  }
  if (body_complete()) {
    fail_callback(CallbackError::invalid_acquire);
    return CURL_RECV_BUFFER_ERROR;
  }
  if (_receive_buffer == nullptr || _slot_ready || _tracker.raw_bytes() == _capacity) {
    return CURL_RECV_BUFFER_AGAIN;
  }
  (void)suggested_size;
  auto* const next      = static_cast<std::byte*>(_receive_buffer) + _tracker.raw_bytes();
  auto const remaining  = _capacity - _tracker.raw_bytes();
  buffer->buffer        = next;
  buffer->length        = remaining;
  buffer->token         = this;
  _loan_outstanding     = true;
  _loan_buffer          = next;
  _loan_capacity        = remaining;
  _loan_body_high_water = 0;
  return CURL_RECV_BUFFER_OK;
}

void CurlDirectReceiveState::release_buffer(curl_recv_buffer const* buffer,
                                            std::size_t used) noexcept
{
  if (buffer == nullptr || !_loan_outstanding || buffer->buffer != _loan_buffer ||
      buffer->length != _loan_capacity || buffer->token != this || used > _loan_capacity ||
      used < _loan_body_high_water || !_tracker.advance_raw(used)) {
    fail_callback(CallbackError::invalid_release);
    return;
  }
  _loan_outstanding               = false;
  _loan_ever_released             = true;
  _loan_buffer                    = nullptr;
  _loan_capacity                  = 0;
  _loan_body_high_water           = 0;
  auto const minimum_receive_size = DirectReceiveSlotPool::minimum_slot_size();
  auto const remaining            = _capacity - _tracker.raw_bytes();
  if (!_callback_failed &&
      (remaining == 0 || (_rotate_before_short_loan && _tracker.raw_bytes() != 0 &&
                          remaining < minimum_receive_size && body_bytes() < _requested_size))) {
    _slot_ready = true;
  }
}

std::size_t CurlDirectReceiveState::write_body(char* data,
                                               std::size_t size,
                                               std::size_t nmemb,
                                               void* userdata) noexcept
{
  auto* self = static_cast<CurlDirectReceiveState*>(userdata);
  if (self == nullptr) { return CURL_WRITEFUNC_ERROR; }
  return self->consume_body(data, size, nmemb);
}

std::size_t CurlDirectReceiveState::consume_body(char* data,
                                                 std::size_t size,
                                                 std::size_t nmemb) noexcept
{
  if (_callback_failed) { return CURL_WRITEFUNC_ERROR; }
  if (size != 0 && nmemb > std::numeric_limits<std::size_t>::max() / size) {
    fail_callback(CallbackError::size_overflow);
    return CURL_WRITEFUNC_ERROR;
  }
  auto const nbytes = size * nmemb;
  if (nbytes == 0) { return 0; }
  if (data == nullptr) {
    fail_callback(CallbackError::invalid_callback_data);
    return CURL_WRITEFUNC_ERROR;
  }
  auto const loan_base       = reinterpret_cast<std::uintptr_t>(_loan_buffer);
  auto const body_ptr        = reinterpret_cast<std::uintptr_t>(data);
  bool const in_current_loan = _loan_outstanding && body_ptr >= loan_base &&
                               body_ptr - loan_base <= _loan_capacity &&
                               nbytes <= _loan_capacity - (body_ptr - loan_base);
  if (!in_current_loan) {
    fail_callback(CallbackError::body_outside_loan);
    return CURL_WRITEFUNC_ERROR;
  }
  _loan_body_high_water =
    std::max(_loan_body_high_water, static_cast<std::size_t>(body_ptr - loan_base) + nbytes);

  auto const disposition = _response.body_disposition(_requested_offset,
                                                      _requested_size,
                                                      _object_snapshot->expected_size(),
                                                      _object_snapshot->requires_entity_tag());
  if (disposition == DirectReceiveBodyDisposition::discard) { return nbytes; }
  if (disposition != DirectReceiveBodyDisposition::accept_range || !accept_response_snapshot()) {
    fail_callback(_response.transfer_encoding_seen() ? CallbackError::transfer_encoding
                                                     : CallbackError::response_not_accepted);
    return CURL_WRITEFUNC_ERROR;
  }
  if (body_bytes() > _requested_size || nbytes > _requested_size - body_bytes()) {
    fail_callback(CallbackError::body_length_exceeded);
    return CURL_WRITEFUNC_ERROR;
  }
  if (!_tracker.record_body(data, nbytes)) {
    fail_callback(_tracker.span_capacity_exhausted() ? CallbackError::span_capacity_exhausted
                                                     : CallbackError::body_outside_loan);
    return CURL_WRITEFUNC_ERROR;
  }
  return nbytes;
}

bool CurlDirectReceiveState::accept_response_snapshot() noexcept
{
  if (_response.body_disposition(_requested_offset,
                                 _requested_size,
                                 _object_snapshot->expected_size(),
                                 _object_snapshot->requires_entity_tag()) !=
      DirectReceiveBodyDisposition::accept_range) {
    return false;
  }
  if (!_snapshot_validation_attempted) {
    _snapshot_validation_attempted = true;
    _snapshot_accepted             = _object_snapshot->accept_entity_tag(_response.entity_tag());
  }
  return _snapshot_accepted;
}

std::size_t CurlDirectReceiveState::write_header(char* data,
                                                 std::size_t size,
                                                 std::size_t nmemb,
                                                 void* userdata) noexcept
{
  auto* self = static_cast<CurlDirectReceiveState*>(userdata);
  if (self == nullptr) { return CURL_WRITEFUNC_ERROR; }
  return self->consume_header(data, size, nmemb);
}

std::size_t CurlDirectReceiveState::consume_header(char* data,
                                                   std::size_t size,
                                                   std::size_t nmemb) noexcept
{
  if (_callback_failed) { return CURL_WRITEFUNC_ERROR; }
  if (size != 0 && nmemb > std::numeric_limits<std::size_t>::max() / size) {
    fail_callback(CallbackError::size_overflow);
    return CURL_WRITEFUNC_ERROR;
  }
  auto const nbytes = size * nmemb;
  if (nbytes == 0) { return 0; }
  if (data == nullptr) {
    fail_callback(CallbackError::invalid_callback_data);
    return CURL_WRITEFUNC_ERROR;
  }
  _response.consume_header(std::string_view{data, nbytes});
  auto const disposition = _response.body_disposition(_requested_offset,
                                                      _requested_size,
                                                      _object_snapshot->expected_size(),
                                                      _object_snapshot->requires_entity_tag());
  if (disposition == DirectReceiveBodyDisposition::accept_range && !accept_response_snapshot()) {
    fail_callback(CallbackError::object_snapshot_mismatch);
    return nbytes;
  }
  if (disposition == DirectReceiveBodyDisposition::reject &&
      _response.is_partial_content_response()) {
    fail_callback(_response.transfer_encoding_seen() ? CallbackError::transfer_encoding
                                                     : CallbackError::response_not_accepted);
    return nbytes;
  }
  return nbytes;
}

void CurlDirectReceiveState::fail_callback(CallbackError error) noexcept
{
  _callback_failed = true;
  _slot_ready      = false;
  if (_callback_error == CallbackError::none) { _callback_error = error; }
}

std::string_view CurlDirectReceiveState::callback_error_message(CallbackError error) noexcept
{
  switch (error) {
    case CallbackError::none: return {};
    case CallbackError::size_overflow: return "receive callback byte count overflow";
    case CallbackError::invalid_callback_data: return "receive callback supplied null data";
    case CallbackError::invalid_acquire: return "invalid or repeated receive-buffer acquisition";
    case CallbackError::invalid_release: return "invalid receive-buffer release";
    case CallbackError::body_outside_loan:
      return "HTTP body was not backed by the loaned receive buffer";
    case CallbackError::body_length_exceeded:
      return "HTTP body exceeded the exact requested range length";
    case CallbackError::span_capacity_exhausted:
      return "direct receive exceeded the bounded body-span capacity";
    case CallbackError::transfer_encoding:
      return "direct receive does not support Transfer-Encoding";
    case CallbackError::response_not_accepted:
      return "direct receive rejected body bytes before exact Range validation";
    case CallbackError::object_snapshot_mismatch:
      return "direct receive response does not match the object version snapshot";
  }
  return "unknown receive callback failure";
}

#endif

}  // namespace detail
}  // namespace kvikio
