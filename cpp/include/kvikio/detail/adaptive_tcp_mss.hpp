/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

#include <curl/curl.h>

namespace kvikio::detail {

struct TcpMssSample {
  std::string origin;  // scheme, host and effective port; never path, credentials or query
  std::string local_address;
  int family{};
  std::uint64_t body_bytes{};
  std::uint32_t path_mtu{};
  std::uint32_t receive_mss{};
  std::string peer_address;
};

enum class TcpMssDecision { UNOBSERVABLE, UNPROVEN, JUMBO, INTERMEDIATE, RETIRE, FALLBACK };

struct TcpMssStats {
  std::uint64_t unobservable{};
  std::uint64_t unproven{};
  std::uint64_t jumbo{};
  std::uint64_t intermediate{};
  std::uint64_t retired{};
  std::uint64_t fallback{};
};

/**
 * Observe completed response bodies, not NIC MTU or HEAD responses. Small-MSS connections are
 * retired only after a jumbo response from the same origin/local address/address family.
 * At most 24 retirements are allowed in a 60-second evidence window. Jumbo observations do not
 * replenish that budget. Expired evidence requires a new jumbo observation. Unknown, intermediate
 * and small responses fail open. Retired peer addresses are temporary connection preferences,
 * never a hard denylist. Neither this policy nor its curl adapter changes DNS or retries successful
 * reads. Thresholds are deliberately conservative heuristics, not throughput estimates.
 */
class AdaptiveTcpMssPolicy {
 public:
  using Clock                                       = std::chrono::steady_clock;
  static constexpr std::uint64_t minimum_body_bytes = 64 * 1024;
  static constexpr unsigned int retirement_budget   = 24;
  static constexpr auto evidence_lifetime           = std::chrono::seconds{60};
  static constexpr std::size_t maximum_origins      = 128;

  TcpMssDecision observe(TcpMssSample const& sample, Clock::time_point now = Clock::now());
  TcpMssStats stats() const;
  void record_unobservable();

  // Before connect the local address is unknown. Only use a peer hint if there is one live
  // local path with jumbo evidence for this origin/family. Hints are not universal MTU facts.
  bool avoid_peer(std::string const& origin,
                  int family,
                  std::string const& peer,
                  Clock::time_point now = Clock::now()) const;
  void allow_fallback(std::string const& origin, Clock::time_point now = Clock::now());

 private:
  using Key = std::tuple<std::string, std::string, int>;
  struct Evidence {
    Clock::time_point expires;
    unsigned int remaining{retirement_budget};
    bool suspended{};
    // At most retirement_budget distinct peers can be added during this evidence window.
    std::set<std::string> peers;
  };
  mutable std::mutex _mutex;
  std::map<Key, Evidence> _evidence;
  TcpMssStats _stats;
};

// Same process-wide evidence/budget for easy-threadpool and all multi-poll reactors.
AdaptiveTcpMssPolicy& adaptive_tcp_mss_policy();

// Throws at opt-in setup if the platform/curl cannot safely make a pre-reuse decision.
void set_up_adaptive_tcp_mss(CURL* easy);

// Per-easy-handle callback state, never shared across simultaneous transfers. The owner must
// destroy it before releasing/resetting its CURL handle. At most one pre-HTTP fail-open attempt
// is allowed, independently of the ordinary HTTP retry budget.
class AdaptiveTcpMssConnection {
 public:
  explicit AdaptiveTcpMssConnection(CURL* easy);
  ~AdaptiveTcpMssConnection() noexcept;
  AdaptiveTcpMssConnection(AdaptiveTcpMssConnection const&)            = delete;
  AdaptiveTcpMssConnection& operator=(AdaptiveTcpMssConnection const&) = delete;
  bool retry_unfiltered(CURLcode result) noexcept;

 private:
  static curl_socket_t open_socket(void*, curlsocktype, curl_sockaddr*) noexcept;
  CURL* _easy;
  std::string _skipped_origin;
  bool _unfiltered{};
};

}  // namespace kvikio::detail
