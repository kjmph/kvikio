/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <kvikio/detail/adaptive_tcp_mss.hpp>
#include <kvikio/shim/libcurl.hpp>

using kvikio::detail::AdaptiveTcpMssPolicy;
using kvikio::detail::TcpMssDecision;
using kvikio::detail::TcpMssSample;

TEST(AdaptiveTcpMss, unsupported_curl_rejects_explicit_opt_in)
{
#ifdef CURL_CONN_REUSE_RETIRE
  GTEST_SKIP() << "This libcurl supports the experimental reuse callback";
#else
  auto handle = kvikio::LibCurl::instance().get_handle();
  EXPECT_THROW(kvikio::detail::set_up_adaptive_tcp_mss(handle.get()), std::runtime_error);
#endif
}

namespace {
auto const now = AdaptiveTcpMssPolicy::Clock::time_point{};
TcpMssSample jumbo()
{
  return {"https://example.test:443", "192.0.2.1", 2, 1024 * 1024, 9000, 8948, "192.0.2.10"};
}
TcpMssSample small()
{
  auto sample         = jumbo();
  sample.path_mtu     = 1500;
  sample.receive_mss  = 1448;
  sample.peer_address = "192.0.2.20";
  return sample;
}
}  // namespace

TEST(AdaptiveTcpMss, requires_observed_jumbo_response)
{
  AdaptiveTcpMssPolicy policy;
  EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::UNPROVEN);
  EXPECT_EQ(policy.observe(jumbo(), now), TcpMssDecision::JUMBO);
  EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::RETIRE);
}

TEST(AdaptiveTcpMss, receive_mss_matters_even_with_jumbo_pmtu)
{
  AdaptiveTcpMssPolicy policy;
  auto sample     = small();
  sample.path_mtu = 9000;
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
  policy.observe(jumbo(), now);
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::RETIRE);
}

TEST(AdaptiveTcpMss, isolates_origin_local_address_and_family)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  auto sample   = small();
  sample.origin = "https://other.test:443";
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
  sample        = small();
  sample.origin = "http://example.test:80";
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
  sample        = small();
  sample.origin = "https://example.test:8443";
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
  sample               = small();
  sample.local_address = "192.0.2.2";
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
  sample        = small();
  sample.family = 10;
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNPROVEN);
}

TEST(AdaptiveTcpMss, small_or_unknown_samples_do_not_create_evidence)
{
  for (int missing = 0; missing < 6; ++missing) {
    AdaptiveTcpMssPolicy policy;
    auto sample = jumbo();
    switch (missing) {
      case 0: sample.body_bytes = AdaptiveTcpMssPolicy::minimum_body_bytes - 1; break;
      case 1: sample.path_mtu = 0; break;
      case 2: sample.receive_mss = 0; break;
      case 3: sample.origin.clear(); break;
      case 4: sample.local_address.clear(); break;
      case 5: sample.family = 0; break;
    }
    EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNOBSERVABLE);
    EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::UNPROVEN);
    policy.observe(jumbo(), now);
    EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::UNOBSERVABLE);
  }
}

TEST(AdaptiveTcpMss, intermediate_mtu_is_kept)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  auto sample        = small();
  sample.path_mtu    = 4500;
  sample.receive_mss = 4400;
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::INTERMEDIATE);
}

TEST(AdaptiveTcpMss, jumbo_samples_cannot_reset_retirement_budget)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  for (unsigned int i = 0; i < AdaptiveTcpMssPolicy::retirement_budget; ++i) {
    EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::RETIRE);
    EXPECT_EQ(policy.observe(jumbo(), now), TcpMssDecision::JUMBO);
  }
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::FALLBACK);
    EXPECT_EQ(policy.observe(jumbo(), now), TcpMssDecision::JUMBO);
  }
  EXPECT_EQ(policy.stats().retired, AdaptiveTcpMssPolicy::retirement_budget);
  EXPECT_EQ(policy.stats().fallback, 100);
}

TEST(AdaptiveTcpMss, expired_evidence_requires_new_proof)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  auto later = now + AdaptiveTcpMssPolicy::evidence_lifetime;
  EXPECT_EQ(policy.observe(small(), later), TcpMssDecision::UNPROVEN);
  EXPECT_EQ(policy.observe(jumbo(), later), TcpMssDecision::JUMBO);
  EXPECT_EQ(policy.observe(small(), later), TcpMssDecision::RETIRE);
}

TEST(AdaptiveTcpMss, bounded_state_does_not_evict_live_budgets)
{
  AdaptiveTcpMssPolicy policy;
  for (std::size_t i = 0; i < AdaptiveTcpMssPolicy::maximum_origins; ++i) {
    auto sample = jumbo();
    sample.origin += std::to_string(i);
    policy.observe(sample, now);
  }
  policy.observe(jumbo(), now);  // Additional origin cannot displace a live entry.
  EXPECT_EQ(policy.observe(small(), now), TcpMssDecision::UNPROVEN);
  auto sample = small();
  sample.origin += "0";
  EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::RETIRE);
  auto later = now + AdaptiveTcpMssPolicy::evidence_lifetime;
  policy.observe(jumbo(), later);
  EXPECT_EQ(policy.observe(small(), later), TcpMssDecision::RETIRE);
}

TEST(AdaptiveTcpMss, concurrent_reactors_share_one_budget)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  std::atomic<unsigned int> retired{};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < 100; ++j) {
        if (policy.observe(small(), now) == TcpMssDecision::RETIRE) { ++retired; }
        policy.observe(jumbo(), now);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(retired, AdaptiveTcpMssPolicy::retirement_budget);
  EXPECT_EQ(policy.stats().retired, retired);
}

TEST(AdaptiveTcpMss, peer_hint_requires_retirement_and_expires)
{
  AdaptiveTcpMssPolicy policy;
  auto sample = small();
  auto avoid  = [&](auto time) {
    return policy.avoid_peer(sample.origin, sample.family, sample.peer_address, time);
  };
  policy.observe(sample, now);
  EXPECT_FALSE(avoid(now));
  policy.observe(jumbo(), now);
  EXPECT_FALSE(avoid(now));
  policy.observe(sample, now);
  EXPECT_TRUE(avoid(now));
  EXPECT_FALSE(avoid(now + AdaptiveTcpMssPolicy::evidence_lifetime));
  policy.observe(jumbo(), now + AdaptiveTcpMssPolicy::evidence_lifetime);
  EXPECT_FALSE(avoid(now + AdaptiveTcpMssPolicy::evidence_lifetime));
}

TEST(AdaptiveTcpMss, peer_hints_are_not_global_ip_denylists)
{
  AdaptiveTcpMssPolicy policy;
  auto sample = small();
  policy.observe(jumbo(), now);
  policy.observe(sample, now);
  EXPECT_FALSE(policy.avoid_peer("https://other.test:443", 2, sample.peer_address, now));
  EXPECT_FALSE(policy.avoid_peer(sample.origin, 10, sample.peer_address, now));
  EXPECT_FALSE(policy.avoid_peer(sample.origin, 2, jumbo().peer_address, now));
  auto other_path          = jumbo();
  other_path.local_address = "192.0.2.2";
  policy.observe(other_path, now);
  EXPECT_FALSE(policy.avoid_peer(sample.origin, 2, sample.peer_address, now));
}

TEST(AdaptiveTcpMss, fallback_suspends_filtering_and_retirement_without_rearming)
{
  AdaptiveTcpMssPolicy policy;
  auto sample = small();
  policy.observe(jumbo(), now);
  policy.observe(sample, now);
  policy.allow_fallback(sample.origin, now);
  for (int i = 0; i < 100; ++i) {
    policy.observe(jumbo(), now);
    EXPECT_FALSE(policy.avoid_peer(sample.origin, 2, sample.peer_address, now));
    EXPECT_EQ(policy.observe(sample, now), TcpMssDecision::FALLBACK);
  }
  auto later = now + AdaptiveTcpMssPolicy::evidence_lifetime;
  policy.observe(jumbo(), later);
  EXPECT_EQ(policy.observe(sample, later), TcpMssDecision::RETIRE);
  EXPECT_TRUE(policy.avoid_peer(sample.origin, 2, sample.peer_address, later));
}

TEST(AdaptiveTcpMss, jumbo_from_previously_small_peer_removes_hint)
{
  AdaptiveTcpMssPolicy policy;
  auto sample = small();
  policy.observe(jumbo(), now);
  policy.observe(sample, now);
  auto improved         = jumbo();
  improved.peer_address = sample.peer_address;
  policy.observe(improved, now);
  EXPECT_FALSE(policy.avoid_peer(sample.origin, 2, sample.peer_address, now));
}

TEST(AdaptiveTcpMss, peer_hint_storage_is_bounded_by_retirement_budget)
{
  AdaptiveTcpMssPolicy policy;
  policy.observe(jumbo(), now);
  for (unsigned int i = 0; i < AdaptiveTcpMssPolicy::retirement_budget + 5; ++i) {
    auto sample = small();
    sample.peer_address += std::to_string(i);
    policy.observe(sample, now);
    EXPECT_EQ(policy.avoid_peer(sample.origin, 2, sample.peer_address, now),
              i < AdaptiveTcpMssPolicy::retirement_budget);
  }
}
