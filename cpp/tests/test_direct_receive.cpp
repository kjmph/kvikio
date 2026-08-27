/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <gtest/gtest.h>

#include <kvikio/detail/direct_receive.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace {

constexpr std::size_t test_object_size = 1'000'000;

std::shared_ptr<kvikio::detail::DirectReceiveObjectSnapshot> generic_http_snapshot(
  std::size_t expected_size = test_object_size)
{
  return std::make_shared<kvikio::detail::DirectReceiveObjectSnapshot>(expected_size, false, false);
}

std::shared_ptr<kvikio::detail::DirectReceiveObjectSnapshot> s3_snapshot(
  std::size_t expected_size = test_object_size)
{
  return std::make_shared<kvikio::detail::DirectReceiveObjectSnapshot>(expected_size, true, true);
}

}  // namespace

TEST(DirectReceiveHostPlacement, copies_validated_framing_spans_after_full_validation)
{
  std::array<std::byte, 16> source{};
  source[2]  = std::byte{0xa1};
  source[3]  = std::byte{0xa2};
  source[4]  = std::byte{0xa3};
  source[9]  = std::byte{0xb1};
  source[10] = std::byte{0xb2};
  std::array<std::byte, 14> destination{};
  destination.fill(std::byte{0x6d});

  kvikio::detail::DirectReceiveReleasedSlot released;
  released.spans[0]   = {.source_offset = 2, .destination_offset = 3, .size = 3};
  released.spans[1]   = {.source_offset = 9, .destination_offset = 6, .size = 2};
  released.span_count = 2;
  released.raw_bytes  = 11;
  released.body_bytes = 5;

  auto const placement = kvikio::detail::place_direct_receive_on_host(
    source.data(), source.size(), released, destination.data(), destination.size(), 3, false);
  EXPECT_EQ(placement.direct_bytes, 0);
  EXPECT_EQ(placement.staged_bytes, 5);
  EXPECT_EQ(destination[2], std::byte{0x6d});
  EXPECT_EQ(destination[3], std::byte{0xa1});
  EXPECT_EQ(destination[4], std::byte{0xa2});
  EXPECT_EQ(destination[5], std::byte{0xa3});
  EXPECT_EQ(destination[6], std::byte{0xb1});
  EXPECT_EQ(destination[7], std::byte{0xb2});
  EXPECT_EQ(destination[8], std::byte{0x6d});
}

TEST(DirectReceiveHostPlacement, validates_every_framing_span_before_mutating_destination)
{
  std::array<std::byte, 16> source{};
  source.fill(std::byte{0xa5});
  std::array<std::byte, 12> destination{};
  destination.fill(std::byte{0x3c});
  auto const before = destination;

  kvikio::detail::DirectReceiveReleasedSlot released;
  released.spans[0]   = {.source_offset = 2, .destination_offset = 3, .size = 3};
  released.spans[1]   = {.source_offset = 15, .destination_offset = 6, .size = 2};
  released.span_count = 2;
  released.raw_bytes  = source.size();
  released.body_bytes = 5;

  EXPECT_THROW(
    static_cast<void>(kvikio::detail::place_direct_receive_on_host(
      source.data(), source.size(), released, destination.data(), destination.size(), 3, false)),
    std::logic_error);
  EXPECT_EQ(destination, before);
}

TEST(DirectReceiveHostPlacement, requires_direct_tail_to_cover_only_contiguous_body_bytes)
{
  std::array<std::byte, 12> destination{};
  destination.fill(std::byte{0x4e});
  auto const before = destination;

  kvikio::detail::DirectReceiveReleasedSlot released;
  released.spans[0]   = {.source_offset = 0, .destination_offset = 4, .size = 4};
  released.span_count = 1;
  released.raw_bytes  = 5;
  released.body_bytes = 4;

  EXPECT_THROW(
    static_cast<void>(kvikio::detail::place_direct_receive_on_host(
      destination.data() + 4, 5, released, destination.data(), destination.size(), 4, true)),
    std::logic_error);
  EXPECT_EQ(destination, before);

  released.raw_bytes   = 4;
  auto const placement = kvikio::detail::place_direct_receive_on_host(
    destination.data() + 4, 4, released, destination.data(), destination.size(), 4, true);
  EXPECT_EQ(placement.direct_bytes, 4);
  EXPECT_EQ(placement.staged_bytes, 0);
  EXPECT_EQ(destination, before);
}

TEST(DirectReceiveSpanTracker, records_body_after_headers_without_copy)
{
  std::array<std::byte, 512> storage{};
  kvikio::detail::DirectReceiveSpanTracker tracker;
  tracker.set_buffer(storage.data(), storage.size());

  // The first 91 bytes represent status and response headers in the same raw receive allocation.
  ASSERT_TRUE(tracker.record_body(storage.data() + 91, 100));
  ASSERT_TRUE(tracker.record_body(storage.data() + 191, 80));
  ASSERT_TRUE(tracker.advance_raw(271));

  ASSERT_EQ(tracker.spans().size(), 1);
  EXPECT_EQ(tracker.spans()[0].source_offset, 91);
  EXPECT_EQ(tracker.spans()[0].destination_offset, 0);
  EXPECT_EQ(tracker.spans()[0].size, 180);
  EXPECT_EQ(tracker.body_bytes(), 180);
  EXPECT_EQ(tracker.raw_bytes(), 271);
}

TEST(DirectReceiveSpanTracker, preserves_multiple_noncontiguous_body_chunks)
{
  std::array<std::byte, 1024> storage{};
  kvikio::detail::DirectReceiveSpanTracker tracker;
  tracker.set_buffer(storage.data(), storage.size());

  ASSERT_TRUE(tracker.record_body(storage.data() + 64, 128));
  // Simulate framing between body spans. Destination offsets must remain dense.
  ASSERT_TRUE(tracker.record_body(storage.data() + 200, 17));
  ASSERT_TRUE(tracker.record_body(storage.data() + 217, 31));

  ASSERT_EQ(tracker.spans().size(), 2);
  EXPECT_EQ(tracker.spans()[0].source_offset, 64);
  EXPECT_EQ(tracker.spans()[0].destination_offset, 0);
  EXPECT_EQ(tracker.spans()[0].size, 128);
  EXPECT_EQ(tracker.spans()[1].source_offset, 200);
  EXPECT_EQ(tracker.spans()[1].destination_offset, 128);
  EXPECT_EQ(tracker.spans()[1].size, 48);
}

TEST(DirectReceiveSpanTracker, rejects_unloaned_and_overflowing_spans)
{
  std::array<std::byte, 128> storage{};
  std::array<std::byte, 1> foreign{};
  kvikio::detail::DirectReceiveSpanTracker tracker;
  tracker.set_buffer(storage.data(), storage.size());

  EXPECT_FALSE(tracker.record_body(foreign.data(), 1));
  EXPECT_FALSE(tracker.record_body(storage.data() + 127, 2));
  EXPECT_FALSE(tracker.advance_raw(129));
  EXPECT_EQ(tracker.body_bytes(), 0);
  EXPECT_EQ(tracker.raw_bytes(), 0);
}

TEST(DirectReceiveSpanTracker, retry_discards_attempt_without_touching_destination)
{
  std::array<std::byte, 256> storage{};
  kvikio::detail::DirectReceiveSpanTracker tracker;
  tracker.set_buffer(storage.data(), storage.size());
  ASSERT_TRUE(tracker.record_body(storage.data() + 32, 100));
  ASSERT_TRUE(tracker.advance_raw(132));

  tracker.reset();
  EXPECT_TRUE(tracker.spans().empty());
  EXPECT_EQ(tracker.body_bytes(), 0);
  EXPECT_EQ(tracker.raw_bytes(), 0);
  ASSERT_TRUE(tracker.record_body(storage.data() + 40, 80));
  EXPECT_EQ(tracker.spans().front().destination_offset, 0);
}

TEST(DirectReceiveSpanTracker, rejects_discontiguous_body_after_fixed_span_capacity)
{
  constexpr auto span_count = kvikio::detail::direct_receive_max_spans_per_slot;
  std::array<std::byte, span_count * 2 + 1> storage{};
  kvikio::detail::DirectReceiveSpanTracker tracker;
  tracker.set_buffer(storage.data(), storage.size());

  for (std::size_t i = 0; i < span_count; ++i) {
    ASSERT_TRUE(tracker.record_body(storage.data() + i * 2, 1)) << "span " << i;
  }
  EXPECT_EQ(tracker.spans().size(), span_count);
  EXPECT_EQ(tracker.body_bytes(), span_count);

  EXPECT_FALSE(tracker.record_body(storage.data() + span_count * 2, 1));
  EXPECT_TRUE(tracker.span_capacity_exhausted());
  EXPECT_EQ(tracker.spans().size(), span_count);
  EXPECT_EQ(tracker.body_bytes(), span_count);
}

TEST(DirectReceiveResponse, accepts_strict_http11_range_response)
{
  kvikio::detail::DirectReceiveResponse response;
  response.consume_header("HTTP/1.1 206 Partial Content\r\n");
  response.consume_header("Content-Length: 4096\r\n");
  response.consume_header("Content-Range: bytes 8192-12287/1000000\r\n");
  response.consume_header("Content-Encoding: identity\r\n");
  EXPECT_EQ(response.body_disposition(8192, 4096, test_object_size, false),
            kvikio::detail::DirectReceiveBodyDisposition::undecided);
  response.consume_header("\r\n");

  // Generic HTTP retains compatibility with exact-range servers that do not provide an ETag.
  EXPECT_EQ(response.body_disposition(8192, 4096, test_object_size, false),
            kvikio::detail::DirectReceiveBodyDisposition::accept_range);
  EXPECT_FALSE(
    response.validate(206, CURL_HTTP_VERSION_1_1, 4096, 8192, 4096, test_object_size, false)
      .has_value());
  EXPECT_FALSE(response.entity_tag().has_value());
}

TEST(DirectReceiveResponse, accepts_duplicate_identical_length_and_concrete_range_total)
{
  kvikio::detail::DirectReceiveResponse response;
  response.consume_header("HTTP/1.1 206 Partial Content\r\n");
  response.consume_header("Content-Length: 32\r\n");
  response.consume_header("Content-Length: 32\r\n");
  response.consume_header("Content-Range: bytes 8-39/100\r\n");
  response.consume_header("\r\n");

  EXPECT_EQ(response.body_disposition(8, 32, 100, false),
            kvikio::detail::DirectReceiveBodyDisposition::accept_range);
  EXPECT_FALSE(response.validate(206, CURL_HTTP_VERSION_1_1, 32, 8, 32, 100, false).has_value());
}

TEST(DirectReceiveResponse, requires_concrete_matching_object_total)
{
  auto disposition = [](std::string_view total, std::size_t expected_object_size) {
    kvikio::detail::DirectReceiveResponse response;
    response.consume_header("HTTP/1.1 206 Partial Content\r\n");
    response.consume_header("Content-Length: 32\r\n");
    auto const range = std::string{"Content-Range: bytes 8-39/"} + std::string{total} + "\r\n";
    response.consume_header(range);
    response.consume_header("\r\n");
    return response.body_disposition(8, 32, expected_object_size, false);
  };

  using Disposition = kvikio::detail::DirectReceiveBodyDisposition;
  EXPECT_EQ(disposition("100", 100), Disposition::accept_range);
  EXPECT_EQ(disposition("*", 100), Disposition::reject);
  EXPECT_EQ(disposition("101", 100), Disposition::reject);
}

TEST(DirectReceiveResponse, requires_one_strong_s3_entity_tag)
{
  auto disposition = [](std::initializer_list<std::string_view> entity_tags) {
    kvikio::detail::DirectReceiveResponse response;
    response.consume_header("HTTP/1.1 206 Partial Content\r\n");
    response.consume_header("Content-Length: 32\r\n");
    response.consume_header("Content-Range: bytes 8-39/100\r\n");
    for (auto const entity_tag : entity_tags) {
      auto const header = std::string{"ETag: "} + std::string{entity_tag} + "\r\n";
      response.consume_header(header);
    }
    response.consume_header("\r\n");
    return response.body_disposition(8, 32, 100, true);
  };

  using Disposition = kvikio::detail::DirectReceiveBodyDisposition;
  EXPECT_EQ(disposition({"\"version-1\""}), Disposition::accept_range);
  EXPECT_EQ(disposition({}), Disposition::reject);
  EXPECT_EQ(disposition({"W/\"version-1\""}), Disposition::reject);
  EXPECT_EQ(disposition({"version-1"}), Disposition::reject);
  EXPECT_EQ(disposition({"\"version-1\"", "\"version-1\""}), Disposition::reject);
}

TEST(DirectReceiveObjectSnapshot, concurrent_ranges_converge_on_one_entity_tag)
{
  constexpr std::size_t num_ranges = 32;
  auto snapshot                    = s3_snapshot(100);
  std::array<bool, num_ranges> accepted{};
  std::atomic<bool> start{};
  std::vector<std::thread> ranges;
  ranges.reserve(num_ranges);

  for (std::size_t i = 0; i < num_ranges; ++i) {
    ranges.emplace_back([&, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto const entity_tag = i % 2 == 0 ? "\"version-1\"" : "\"version-2\"";
      accepted[i]           = snapshot->accept_entity_tag(entity_tag);
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& range : ranges) {
    range.join();
  }

  auto const selected = snapshot->if_match_entity_tag();
  ASSERT_TRUE(selected.has_value());
  ASSERT_TRUE(selected == "\"version-1\"" || selected == "\"version-2\"");
  for (std::size_t i = 0; i < num_ranges; ++i) {
    auto const entity_tag = i % 2 == 0 ? "\"version-1\"" : "\"version-2\"";
    EXPECT_EQ(accepted[i], entity_tag == selected.value()) << "range " << i;
  }
}

TEST(DirectReceiveResponse, discards_redirect_body_then_accepts_final_range)
{
  kvikio::detail::DirectReceiveResponse response;
  response.consume_header("HTTP/1.1 302 Found\r\n");
  response.consume_header("Transfer-Encoding: chunked\r\n");
  response.consume_header("Content-Encoding: gzip\r\n");
  response.consume_header("\r\n");
  EXPECT_EQ(response.body_disposition(8192, 32, test_object_size, false),
            kvikio::detail::DirectReceiveBodyDisposition::discard);

  response.consume_header("HTTP/1.1 206 Partial Content\r\n");
  response.consume_header("Content-Length: 32\r\n");
  response.consume_header("Content-Range: bytes 8192-8223/1000000\r\n");
  response.consume_header("\r\n");
  EXPECT_EQ(response.body_disposition(8192, 32, test_object_size, false),
            kvikio::detail::DirectReceiveBodyDisposition::accept_range);
}

TEST(DirectReceiveResponse, discards_informational_body_then_accepts_final_range)
{
  kvikio::detail::DirectReceiveResponse response;
  response.consume_header("HTTP/1.1 103 Early Hints\r\n");
  response.consume_header("Link: </object>; rel=preload\r\n");
  response.consume_header("\r\n");
  EXPECT_EQ(response.body_disposition(8, 32, 100, false),
            kvikio::detail::DirectReceiveBodyDisposition::discard);

  response.consume_header("HTTP/1.1 206 Partial Content\r\n");
  response.consume_header("Content-Length: 32\r\n");
  response.consume_header("Content-Range: bytes 8-39/100\r\n");
  response.consume_header("\r\n");
  EXPECT_EQ(response.body_disposition(8, 32, 100, false),
            kvikio::detail::DirectReceiveBodyDisposition::accept_range);
}

TEST(DirectReceiveResponse, rejects_authentication_challenge_bodies)
{
  using Disposition = kvikio::detail::DirectReceiveBodyDisposition;
  for (auto const status : {std::string_view{"HTTP/1.1 401 Unauthorized\r\n"},
                            std::string_view{"HTTP/1.1 407 Proxy Authentication Required\r\n"}}) {
    kvikio::detail::DirectReceiveResponse response;
    response.consume_header(status);
    response.consume_header("Content-Length: 32\r\n");
    response.consume_header("\r\n");
    EXPECT_EQ(response.body_disposition(8, 32, 100, false), Disposition::reject);
  }
}

TEST(DirectReceiveResponse, rejects_malformed_status_headers_and_range_total)
{
  auto disposition = [](std::initializer_list<std::string_view> lines) {
    kvikio::detail::DirectReceiveResponse response;
    for (auto line : lines) {
      response.consume_header(line);
    }
    return response.body_disposition(8, 32, 100, false);
  };

  using Disposition = kvikio::detail::DirectReceiveBodyDisposition;
  EXPECT_EQ(disposition({"HTTP/1.1 0206 Partial Content\r\n", "\r\n"}), Disposition::reject);
  EXPECT_EQ(disposition({" HTTP/1.1 206 Partial Content\r\n", "\r\n"}), Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n", " folded\r\n", "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n", "not-a-header\r\n", "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes 8-39/39\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes8-39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range : bytes 8-39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length\t: 32\r\n",
                         "Content-Range: bytes 8-39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes 8 -39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes 8- 39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Length: 31\r\n",
                         "Content-Range: bytes 8-39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes 8-39/100\r\n",
                         "Content-Range: bytes 8-39/100\r\n",
                         "\r\n"}),
            Disposition::reject);
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n",
                         "Content-Length: 32\r\n",
                         "Content-Range: bytes 8-18446744073709551615/*\r\n",
                         "\r\n"}),
            Disposition::reject);

  std::string const non_ascii_header{"\xE9: value\r\n"};
  EXPECT_EQ(disposition({"HTTP/1.1 206 Partial Content\r\n", non_ascii_header, "\r\n"}),
            Disposition::reject);
}

TEST(DirectReceiveResponse, rejects_full_chunked_or_encoded_responses)
{
  struct Case {
    std::string extra_header;
    long response_code;
  };
  for (auto const& test_case : {Case{"Transfer-Encoding: chunked\r\n", 206},
                                Case{"Content-Encoding: gzip\r\n", 206},
                                Case{"", 200}}) {
    kvikio::detail::DirectReceiveResponse response;
    response.consume_header("HTTP/1.1 206 Partial Content\r\n");
    response.consume_header("Content-Length: 32\r\n");
    response.consume_header("Content-Range: bytes 8-39/100\r\n");
    if (!test_case.extra_header.empty()) { response.consume_header(test_case.extra_header); }
    response.consume_header("\r\n");
    EXPECT_TRUE(
      response.validate(test_case.response_code, CURL_HTTP_VERSION_1_1, 32, 8, 32, 100, false)
        .has_value());
  }
}

TEST(DirectReceiveResponse, rejects_http2_and_range_metadata_mismatches)
{
  auto make_response = [] {
    kvikio::detail::DirectReceiveResponse response;
    response.consume_header("HTTP/1.1 206 Partial Content\r\n");
    response.consume_header("Content-Length: 32\r\n");
    response.consume_header("Content-Range: bytes 8-39/100\r\n");
    response.consume_header("\r\n");
    return response;
  };

  EXPECT_TRUE(
    make_response().validate(206, CURL_HTTP_VERSION_2_0, 32, 8, 32, 100, false).has_value());
  EXPECT_TRUE(
    make_response().validate(206, CURL_HTTP_VERSION_1_1, 31, 8, 32, 100, false).has_value());
  EXPECT_TRUE(
    make_response().validate(206, CURL_HTTP_VERSION_1_1, 32, 9, 32, 100, false).has_value());

  auto duplicate_range = make_response();
  duplicate_range.consume_header("Content-Range: bytes 8-39/100\r\n");
  EXPECT_TRUE(
    duplicate_range.validate(206, CURL_HTTP_VERSION_1_1, 32, 8, 32, 100, false).has_value());
}

TEST(DirectReceiveFallback, only_pre_handoff_capability_failures_are_eligible)
{
#if defined(CURL_HAS_KTLS_DIRECT_RX)
  EXPECT_TRUE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_UNAVAILABLE, 0));

  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    true, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_UNAVAILABLE, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_NONE, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_REQUESTED, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_ACTIVE, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_NOT_BUILT_IN, CURL_KTLS_DIRECT_RX_UNAVAILABLE, 1));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_ABORTED_BY_CALLBACK, CURL_KTLS_DIRECT_RX_UNAVAILABLE, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_SSL_CONNECT_ERROR, CURL_KTLS_DIRECT_RX_REQUESTED, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_PEER_FAILED_VERIFICATION, CURL_KTLS_DIRECT_RX_REQUESTED, 0));
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(
    false, CURLE_OPERATION_TIMEDOUT, CURL_KTLS_DIRECT_RX_REQUESTED, 0));
#else
  EXPECT_FALSE(kvikio::detail::direct_receive_can_fallback(false, CURLE_NOT_BUILT_IN, 0, 0));
#endif
}

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)

static_assert(!std::is_copy_constructible_v<kvikio::detail::CurlDirectReceiveState>);
static_assert(!std::is_copy_assignable_v<kvikio::detail::CurlDirectReceiveState>);
static_assert(!std::is_move_constructible_v<kvikio::detail::CurlDirectReceiveState>);
static_assert(!std::is_move_assignable_v<kvikio::detail::CurlDirectReceiveState>);

namespace {

std::vector<std::byte> receive_storage(std::size_t minimum_size = 0)
{
  return std::vector<std::byte>{
    std::max(minimum_size, kvikio::detail::DirectReceiveSlotPool::minimum_slot_size())};
}

void accept_range(kvikio::detail::CurlDirectReceiveState& state,
                  std::size_t offset,
                  std::size_t size,
                  std::size_t expected_object_size = test_object_size,
                  std::string_view entity_tag      = {})
{
  auto status = std::string{"HTTP/1.1 206 Partial Content\r\n"};
  auto length = std::string{"Content-Length: "} + std::to_string(size) + "\r\n";
  auto range  = std::string{"Content-Range: bytes "} + std::to_string(offset) + "-" +
               std::to_string(offset + size - 1) + "/" + std::to_string(expected_object_size) +
               "\r\n";
  auto end = std::string{"\r\n"};
  ASSERT_EQ(state.consume_header(status.data(), 1, status.size()), status.size());
  ASSERT_EQ(state.consume_header(length.data(), 1, length.size()), length.size());
  ASSERT_EQ(state.consume_header(range.data(), 1, range.size()), range.size());
  if (!entity_tag.empty()) {
    auto header = std::string{"ETag: "} + std::string{entity_tag} + "\r\n";
    ASSERT_EQ(state.consume_header(header.data(), 1, header.size()), header.size());
  }
  ASSERT_EQ(state.consume_header(end.data(), 1, end.size()), end.size());
}

}  // namespace

TEST(CurlDirectReceiveState, loans_consecutive_regions_across_receive_cycles)
{
  auto storage = receive_storage(2 * kvikio::detail::DirectReceiveSlotPool::minimum_slot_size());
  kvikio::detail::CurlDirectReceiveState state{1000, 256, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());

  curl_recv_buffer first{};
  ASSERT_EQ(state.acquire_buffer(128, &first), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(first.buffer, storage.data());
  EXPECT_EQ(first.length, storage.size());
  EXPECT_NE(first.token, nullptr);

  accept_range(state, 1000, 256);
  // The first 20 raw bytes model response headers delivered from this same loan.
  EXPECT_EQ(state.consume_body(static_cast<char*>(first.buffer) + 20, 1, 80), 80);
  state.release_buffer(&first, 100);
  EXPECT_FALSE(state.callback_failed());
  EXPECT_EQ(state.raw_bytes(), 100);

  curl_recv_buffer second{};
  ASSERT_EQ(state.acquire_buffer(200, &second), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(second.buffer, storage.data() + 100);
  EXPECT_EQ(second.length, storage.size() - 100);
  EXPECT_EQ(second.token, first.token);
  EXPECT_EQ(state.consume_body(static_cast<char*>(second.buffer), 1, 176), 176);
  state.release_buffer(&second, 176);
  EXPECT_FALSE(state.callback_failed());
  EXPECT_EQ(state.raw_bytes(), 276);
  EXPECT_EQ(state.body_bytes(), 256);
  state.finalize_current_slot();
  ASSERT_TRUE(state.slot_ready());
  auto const released = state.take_released_slot();
  ASSERT_EQ(released.span_count, 1);
  EXPECT_EQ(released.spans[0].source_offset, 20);
  EXPECT_EQ(released.spans[0].destination_offset, 0);
  EXPECT_EQ(released.spans[0].size, 256);
}

TEST(CurlDirectReceiveState, validates_requested_range_and_slot_installation)
{
  EXPECT_THROW(kvikio::detail::CurlDirectReceiveState(0, 0, generic_http_snapshot()),
               std::invalid_argument);
  EXPECT_THROW(kvikio::detail::CurlDirectReceiveState(
                 std::numeric_limits<std::size_t>::max(), 2, generic_http_snapshot()),
               std::invalid_argument);
  EXPECT_THROW(kvikio::detail::CurlDirectReceiveState(0, 1, nullptr), std::invalid_argument);
  EXPECT_THROW(kvikio::detail::CurlDirectReceiveState(9, 2, generic_http_snapshot(10)),
               std::invalid_argument);

  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot(1)};
  auto storage = receive_storage();
  EXPECT_THROW(state.install_buffer(nullptr, storage.size()), std::logic_error);
  state.install_buffer(storage.data(), storage.size());
  EXPECT_THROW(state.install_buffer(storage.data(), storage.size()), std::logic_error);
}

TEST(CurlDirectReceiveState, shared_s3_snapshot_accepts_v1_v1_convergence)
{
  auto snapshot = s3_snapshot(100);
  kvikio::detail::CurlDirectReceiveState first{0, 32, snapshot};
  kvikio::detail::CurlDirectReceiveState second{32, 32, snapshot};

  accept_range(first, 0, 32, 100, "\"version-1\"");
  accept_range(second, 32, 32, 100, "\"version-1\"");

  EXPECT_TRUE(first.response_body_accepted());
  EXPECT_TRUE(second.response_body_accepted());
  ASSERT_TRUE(snapshot->if_match_entity_tag().has_value());
  EXPECT_EQ(snapshot->if_match_entity_tag().value(), "\"version-1\"");
}

TEST(CurlDirectReceiveState, shared_s3_snapshot_rejects_v1_v2_mismatch)
{
  auto snapshot = s3_snapshot(100);
  kvikio::detail::CurlDirectReceiveState first{0, 32, snapshot};
  kvikio::detail::CurlDirectReceiveState changed{32, 32, snapshot};

  accept_range(first, 0, 32, 100, "\"version-1\"");
  accept_range(changed, 32, 32, 100, "\"version-2\"");

  EXPECT_TRUE(first.response_body_accepted());
  EXPECT_FALSE(changed.response_body_accepted());
  EXPECT_TRUE(changed.callback_failed());
  EXPECT_TRUE(changed.callback_protocol_validation_failed());
  EXPECT_EQ(changed.callback_error(),
            "direct receive response does not match the object version snapshot");
  ASSERT_TRUE(snapshot->if_match_entity_tag().has_value());
  EXPECT_EQ(snapshot->if_match_entity_tag().value(), "\"version-1\"");
}

TEST(CurlDirectReceiveState, rejects_missing_s3_entity_tag_at_header_completion)
{
  kvikio::detail::CurlDirectReceiveState state{0, 32, s3_snapshot(100)};

  accept_range(state, 0, 32, 100);

  EXPECT_TRUE(state.callback_failed());
  EXPECT_TRUE(state.callback_protocol_validation_failed());
  EXPECT_FALSE(state.response_body_accepted());
}

TEST(CurlDirectReceiveState, zero_byte_release_preserves_the_ownership_fence)
{
  auto storage = receive_storage(2 * kvikio::detail::DirectReceiveSlotPool::minimum_slot_size());
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());

  curl_recv_buffer first{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &first), CURL_RECV_BUFFER_OK);
  state.release_buffer(&first, 0);
  EXPECT_FALSE(state.callback_failed());
  EXPECT_EQ(state.raw_bytes(), 0);

  curl_recv_buffer second{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &second), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(second.buffer, first.buffer);
  accept_range(state, 0, 1);
  ASSERT_EQ(state.consume_body(static_cast<char*>(second.buffer), 1, 1), 1);
  state.release_buffer(&second, 1);
  state.finalize_current_slot();
  EXPECT_TRUE(state.slot_ready());
}

TEST(CurlDirectReceiveState, rejects_overflow_and_null_callback_data)
{
  auto make_state = [] {
    auto storage = receive_storage();
    auto state =
      std::make_unique<kvikio::detail::CurlDirectReceiveState>(0, 1, generic_http_snapshot());
    state->install_buffer(storage.data(), storage.size());
    return std::pair{std::move(storage), std::move(state)};
  };

  {
    auto [storage, state] = make_state();
    curl_recv_buffer loan{};
    ASSERT_EQ(state->acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
    EXPECT_EQ(state->consume_body(
                static_cast<char*>(loan.buffer), std::numeric_limits<std::size_t>::max(), 2),
              CURL_WRITEFUNC_ERROR);
    EXPECT_EQ(state->callback_error(), "receive callback byte count overflow");
  }
  {
    auto [storage, state] = make_state();
    EXPECT_EQ(state->consume_header(nullptr, std::numeric_limits<std::size_t>::max(), 2),
              CURL_WRITEFUNC_ERROR);
    EXPECT_EQ(state->callback_error(), "receive callback byte count overflow");
  }
  {
    auto [storage, state] = make_state();
    curl_recv_buffer loan{};
    ASSERT_EQ(state->acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
    EXPECT_EQ(state->consume_body(nullptr, 1, 1), CURL_WRITEFUNC_ERROR);
    EXPECT_EQ(state->callback_error(), "receive callback supplied null data");
  }
  {
    auto [storage, state] = make_state();
    EXPECT_EQ(state->consume_header(nullptr, 1, 1), CURL_WRITEFUNC_ERROR);
    EXPECT_EQ(state->callback_error(), "receive callback supplied null data");
  }
  {
    auto [storage, state] = make_state();
    EXPECT_EQ(state->consume_body(nullptr, 0, 1), 0);
    EXPECT_EQ(state->consume_header(nullptr, 1, 0), 0);
    EXPECT_FALSE(state->callback_failed());
  }
}

TEST(CurlDirectReceiveState, configures_the_dedicated_strict_rx_option)
{
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  auto curl = create_curl_handle();

  // Configure an unrelated SSL option first. Direct receive uses its dedicated boolean option and
  // therefore does not replace the CURLOPT_SSL_OPTIONS mask.
  curl.setopt(CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_NO_REVOKE));
  EXPECT_NO_THROW(state.configure(curl, false));
  EXPECT_NO_THROW(state.configure(curl, true));
}

TEST(CurlDirectReceiveState, rejects_repeated_acquisition)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());

  curl_recv_buffer loan{};
  // suggested_size is advisory. A smaller nonempty tail is still a valid loan.
  EXPECT_EQ(state.acquire_buffer(storage.size() + 1, &loan), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(loan.length, storage.size());
  curl_recv_buffer overlapping{};
  EXPECT_EQ(state.acquire_buffer(1, &overlapping), CURL_RECV_BUFFER_ERROR);
  EXPECT_TRUE(state.callback_failed());

  state.release_buffer(&loan, 0);
}

TEST(CurlDirectReceiveState, rejects_null_acquisition_descriptor)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());

  EXPECT_EQ(state.acquire_buffer(1, nullptr), CURL_RECV_BUFFER_ERROR);
  EXPECT_TRUE(state.callback_failed());
}

TEST(CurlDirectReceiveState, rejects_mismatched_release)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());

  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(1, &loan), CURL_RECV_BUFFER_OK);
  auto wrong  = loan;
  wrong.token = nullptr;
  state.release_buffer(&wrong, 1);
  EXPECT_TRUE(state.callback_failed());
}

TEST(CurlDirectReceiveState, rotates_released_slots_with_global_destination_offsets)
{
  auto first_storage = receive_storage(32 * 1024);
  kvikio::detail::CurlDirectReceiveState state{10, 8, generic_http_snapshot()};
  state.install_buffer(first_storage.data(), first_storage.size());

  curl_recv_buffer first{};
  ASSERT_EQ(state.acquire_buffer(64, &first), CURL_RECV_BUFFER_OK);
  accept_range(state, 10, 8);
  ASSERT_EQ(state.consume_body(static_cast<char*>(first.buffer), 1, 4), 4);
  constexpr std::size_t first_raw_bytes = 24 * 1024;
  state.release_buffer(&first, first_raw_bytes);
  ASSERT_TRUE(state.slot_ready());

  curl_recv_buffer unavailable{};
  EXPECT_EQ(state.acquire_buffer(64, &unavailable), CURL_RECV_BUFFER_AGAIN);

  auto const released_first = state.take_released_slot();
  ASSERT_EQ(released_first.span_count, 1);
  EXPECT_EQ(released_first.body_bytes, 4);
  EXPECT_EQ(released_first.spans[0].destination_offset, 0);
  EXPECT_TRUE(state.needs_buffer());

  auto final_storage = receive_storage();
  state.install_buffer(final_storage.data(), final_storage.size());
  curl_recv_buffer second{};
  ASSERT_EQ(state.acquire_buffer(64, &second), CURL_RECV_BUFFER_OK);
  ASSERT_EQ(state.consume_body(static_cast<char*>(second.buffer), 1, 4), 4);
  state.release_buffer(&second, 4);
  EXPECT_FALSE(state.slot_ready());
  state.finalize_current_slot();
  ASSERT_TRUE(state.slot_ready());

  auto const released_second = state.take_released_slot();
  ASSERT_EQ(released_second.span_count, 1);
  EXPECT_EQ(released_second.body_bytes, 4);
  EXPECT_EQ(released_second.spans[0].destination_offset, 4);
  EXPECT_EQ(state.body_bytes(), 8);
  EXPECT_EQ(state.raw_bytes(), first_raw_bytes + 4);
  EXPECT_FALSE(state.needs_buffer());
}

TEST(CurlDirectReceiveState, exact_full_final_slot_does_not_wait_for_another_buffer)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{
    0, storage.size(), generic_http_snapshot(storage.size())};
  state.install_buffer(storage.data(), storage.size());

  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  accept_range(state, 0, storage.size(), storage.size());
  ASSERT_EQ(state.consume_body(static_cast<char*>(loan.buffer), 1, storage.size()), storage.size());
  state.release_buffer(&loan, storage.size());
  ASSERT_TRUE(state.slot_ready());

  auto const released = state.take_released_slot();
  EXPECT_EQ(released.body_bytes, storage.size());
  EXPECT_TRUE(state.body_complete());
  EXPECT_FALSE(state.needs_buffer());

  curl_recv_buffer unexpected{};
  EXPECT_EQ(state.acquire_buffer(1, &unexpected), CURL_RECV_BUFFER_ERROR);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_FALSE(state.needs_buffer());
}

TEST(CurlDirectReceiveState, short_direct_loans_reuse_zero_byte_releases)
{
  std::array<std::byte, 1> storage{};
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  EXPECT_THROW(state.install_buffer(storage.data(), 0), std::logic_error);

  // GPU pool configuration retains its TLS-record floor, while a validated host tail may be
  // smaller than one record. A socket EAGAIN can release that loan without progress; retain and
  // reacquire the same byte instead of rotating the reactor through an empty ready slot.
  EXPECT_THROW(state.install_buffer(storage.data(), storage.size()), std::logic_error);
  state.install_buffer(storage.data(), storage.size(), false);
  curl_recv_buffer loan{};
  EXPECT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(loan.buffer, storage.data());
  EXPECT_EQ(loan.length, storage.size());
  accept_range(state, 0, 1);
  state.release_buffer(&loan, 0);
  EXPECT_FALSE(state.slot_ready());
  EXPECT_FALSE(state.needs_buffer());

  curl_recv_buffer retry_loan{};
  EXPECT_EQ(state.acquire_buffer(storage.size(), &retry_loan), CURL_RECV_BUFFER_OK);
  EXPECT_EQ(retry_loan.buffer, storage.data());
  EXPECT_EQ(retry_loan.length, storage.size());
  EXPECT_EQ(state.consume_body(static_cast<char*>(retry_loan.buffer), 1, 1), 1);
  state.release_buffer(&retry_loan, 1);
  EXPECT_TRUE(state.slot_ready());
}

TEST(CurlDirectReceiveState, release_cannot_exclude_a_retained_body_span)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 5, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(32, &loan), CURL_RECV_BUFFER_OK);
  accept_range(state, 0, 5);
  EXPECT_EQ(state.consume_body(static_cast<char*>(loan.buffer) + 10, 1, 5), 5);

  state.release_buffer(&loan, 14);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_EQ(state.raw_bytes(), 0);
}

TEST(CurlDirectReceiveState, rejects_transfer_encoding_before_body_handoff)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot(100)};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  for (auto header : {std::string{"HTTP/1.1 206 Partial Content\r\n"},
                      std::string{"Content-Length: 1\r\n"},
                      std::string{"Content-Range: bytes 0-0/100\r\n"},
                      std::string{"Transfer-Encoding: chunked\r\n"},
                      std::string{"\r\n"}}) {
    EXPECT_EQ(state.consume_header(header.data(), 1, header.size()), header.size());
  }
  EXPECT_EQ(state.consume_body(static_cast<char*>(loan.buffer), 1, 1), CURL_WRITEFUNC_ERROR);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_EQ(state.callback_error(), "direct receive does not support Transfer-Encoding");

  // Once a callback fails, subsequent callbacks remain failed without replacing the first error.
  EXPECT_EQ(state.consume_body(reinterpret_cast<char*>(storage.data()), 1, 1),
            CURL_WRITEFUNC_ERROR);
  EXPECT_EQ(state.callback_error(), "direct receive does not support Transfer-Encoding");
}

TEST(CurlDirectReceiveState, callback_failure_never_publishes_a_slot)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  accept_range(state, 0, 1);

  EXPECT_EQ(state.consume_body(static_cast<char*>(loan.buffer), 1, 2), CURL_WRITEFUNC_ERROR);
  ASSERT_TRUE(state.callback_failed());
  state.release_buffer(&loan, storage.size());

  EXPECT_FALSE(state.slot_ready());
  EXPECT_FALSE(state.needs_buffer());
  EXPECT_EQ(state.callback_error(), "HTTP body exceeded the exact requested range length");
  EXPECT_TRUE(state.callback_protocol_validation_failed());
  EXPECT_THROW(static_cast<void>(state.take_released_slot()), std::logic_error);
}

TEST(CurlDirectReceiveState, discarded_body_still_obeys_loan_release_fence)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{0, 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  for (auto header : {std::string{"HTTP/1.1 302 Found\r\n"},
                      std::string{"Content-Length: 4\r\n"},
                      std::string{"\r\n"}}) {
    ASSERT_EQ(state.consume_header(header.data(), 1, header.size()), header.size());
  }
  ASSERT_EQ(state.consume_body(static_cast<char*>(loan.buffer) + 10, 1, 4), 4);
  state.release_buffer(&loan, 13);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_EQ(state.body_bytes(), 0);
}

TEST(CurlDirectReceiveState, rejects_body_callback_when_fixed_span_capacity_is_exhausted)
{
  constexpr auto span_count = kvikio::detail::direct_receive_max_spans_per_slot;
  auto storage              = receive_storage(span_count * 2 + 1);
  kvikio::detail::CurlDirectReceiveState state{0, span_count + 1, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);
  accept_range(state, 0, span_count + 1);

  for (std::size_t i = 0; i < span_count; ++i) {
    ASSERT_EQ(state.consume_body(reinterpret_cast<char*>(storage.data() + i * 2), 1, 1), 1)
      << "span " << i;
  }
  EXPECT_EQ(state.consume_body(reinterpret_cast<char*>(storage.data() + span_count * 2), 1, 1),
            CURL_WRITEFUNC_ERROR);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_EQ(state.callback_error(), "direct receive exceeded the bounded body-span capacity");
  EXPECT_EQ(state.body_bytes(), span_count);
}

TEST(CurlDirectReceiveState, rejects_body_before_exact_range_headers_complete)
{
  auto storage = receive_storage();
  kvikio::detail::CurlDirectReceiveState state{8, 4, generic_http_snapshot()};
  state.install_buffer(storage.data(), storage.size());
  curl_recv_buffer loan{};
  ASSERT_EQ(state.acquire_buffer(storage.size(), &loan), CURL_RECV_BUFFER_OK);

  EXPECT_EQ(state.consume_body(static_cast<char*>(loan.buffer), 1, 4), CURL_WRITEFUNC_ERROR);
  EXPECT_TRUE(state.callback_failed());
  EXPECT_EQ(state.callback_error(),
            "direct receive rejected body bytes before exact Range validation");
  EXPECT_EQ(state.body_bytes(), 0);
}

#endif
