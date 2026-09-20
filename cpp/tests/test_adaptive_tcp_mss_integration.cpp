/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/adaptive_tcp_mss.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

#include "utils/env.hpp"

namespace {

// Only per-socket options are changed. No host MTU, sysctl, DNS or external server is needed.
class SmallMssServer {
 public:
  explicit SmallMssServer(std::size_t body_size,
                          std::string const& ip = "127.0.0.1",
                          int mss               = 1024,
                          unsigned short port   = 0)
    : _body(body_size, 'x')
  {
    _listener = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (_listener < 0) { throw std::runtime_error("socket"); }
    try {
      if (setsockopt(_listener, IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss))) {
        throw std::runtime_error("TCP_MAXSEG");
      }
      sockaddr_in address{};
      address.sin_family = AF_INET;
      inet_pton(AF_INET, ip.c_str(), &address.sin_addr);
      address.sin_port = htons(port);
      socklen_t length = sizeof(address);
      if (bind(_listener, reinterpret_cast<sockaddr*>(&address), length) ||
          getsockname(_listener, reinterpret_cast<sockaddr*>(&address), &length) ||
          listen(_listener, 8)) {
        throw std::runtime_error("listen");
      }
      _port   = ntohs(address.sin_port);
      _thread = std::thread([this] { accept_loop(); });
    } catch (...) {
      close(_listener);
      throw;
    }
  }
  ~SmallMssServer()
  {
    _stop = true;
    _thread.join();
    close(_listener);
  }
  std::string origin() const { return "http://127.0.0.1:" + std::to_string(_port); }
  unsigned short port() const { return _port; }
  std::atomic<unsigned int> connections{};
  std::atomic<unsigned int> requests{};
  std::atomic<unsigned int> status{206};
  std::atomic<bool> truncate_body{false};

 private:
  void serve(int fd)
  {
    timeval timeout{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    std::string pending;
    while (!_stop) {
      pollfd event{fd, POLLIN, 0};
      if (poll(&event, 1, 20) <= 0) { continue; }
      std::array<char, 4096> buffer;
      auto count = recv(fd, buffer.data(), buffer.size(), 0);
      if (count <= 0) { break; }
      pending.append(buffer.data(), static_cast<std::size_t>(count));
      if (pending.find("\r\n\r\n") == std::string::npos) { continue; }
      ++requests;
      auto response = "HTTP/1.1 " + std::to_string(status.load()) +
                      " Test\r\nContent-Length: " + std::to_string(_body.size()) +
                      "\r\nContent-Range: bytes 0-" + std::to_string(_body.size() - 1) + "/" +
                      std::to_string(_body.size()) + "\r\nConnection: keep-alive\r\n\r\n" + _body;
      bool const truncated = truncate_body;
      if (truncated) { response.resize(response.size() - _body.size() / 2); }
      std::size_t sent = 0;
      while (sent < response.size() && !_stop) {
        count = send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) { continue; }
        if (count <= 0) { break; }
        sent += static_cast<std::size_t>(count);
      }
      if (truncated || sent != response.size()) { break; }
      pending.clear();
    }
    close(fd);
  }
  void accept_loop()
  {
    std::vector<std::thread> clients;
    while (!_stop) {
      pollfd event{_listener, POLLIN, 0};
      if (poll(&event, 1, 20) <= 0) { continue; }
      int fd = accept4(_listener, nullptr, nullptr, SOCK_CLOEXEC);
      if (fd < 0) { continue; }
      ++connections;
      clients.emplace_back([this, fd] { serve(fd); });
    }
    for (auto& client : clients) {
      client.join();
    }
  }
  int _listener{-1};
  unsigned short _port{};
  std::string _body;
  std::atomic<bool> _stop{};
  std::thread _thread;
};

class AdaptiveTcpMssIntegration : public testing::TestWithParam<kvikio::RemoteIOBackend> {
 protected:
  void SetUp() override
  {
#ifndef CURL_CONN_REUSE_RETIRE
    if (kvikio::defaults::remote_adaptive_tcp_mss()) {
      GTEST_SKIP() << "This libcurl lacks the experimental reuse callback";
    }
#endif
    previous_backend_  = kvikio::defaults::remote_io_backend();
    previous_attempts_ = kvikio::defaults::http_max_attempts();
  }

  void TearDown() override
  {
    kvikio::defaults::set_remote_io_backend(previous_backend_);
    kvikio::defaults::set_http_max_attempts(previous_attempts_);
  }

 private:
  kvikio::RemoteIOBackend previous_backend_{kvikio::defaults::remote_io_backend()};
  std::size_t previous_attempts_{kvikio::defaults::http_max_attempts()};
};

void check_reads(kvikio::RemoteIOBackend backend,
                 bool prove_jumbo,
                 std::size_t size,
                 unsigned int reads = 2)
{
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  SmallMssServer server{size};
  auto& policy = kvikio::detail::adaptive_tcp_mss_policy();
  if (prove_jumbo) {
    // Synthetic prior evidence; the completed reads below use real small-MSS TCP connections.
    policy.observe({server.origin(), "127.0.0.1", AF_INET, 1024 * 1024, 9000, 8948});
  }
  auto const before           = policy.stats();
  auto const previous_backend = kvikio::defaults::remote_io_backend();
  kvikio::defaults::set_remote_io_backend(backend);
  auto remote =
    kvikio::RemoteHandle{std::make_unique<kvikio::HttpEndpoint>(server.origin() + "/data"), size};
  std::vector<char> buffer(size);
  for (unsigned int i = 0; i < reads; ++i) {
    std::fill(buffer.begin(), buffer.end(), '\0');
    auto future = remote.pread(buffer.data(), buffer.size(), 0, buffer.size());
    EXPECT_EQ(future.get(), size);
    EXPECT_TRUE(std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == 'x'; }));
  }
  kvikio::defaults::set_remote_io_backend(previous_backend);
  bool const retire = kvikio::defaults::remote_adaptive_tcp_mss() && prove_jumbo &&
                      size >= kvikio::detail::AdaptiveTcpMssPolicy::minimum_body_bytes;
  EXPECT_EQ(server.connections, retire ? std::min(reads, 2U) : 1U);
  EXPECT_EQ(server.requests, reads);  // A completed response must never be replayed.
  EXPECT_EQ(policy.stats().retired - before.retired, retire ? 1U : 0U);
  EXPECT_EQ(policy.stats().fallback - before.fallback, retire ? reads - 1 : 0U);
}

TEST_P(AdaptiveTcpMssIntegration, retires_only_with_prior_evidence)
{
  check_reads(GetParam(), true, 256 * 1024);
}
TEST_P(AdaptiveTcpMssIntegration, unproven_path_is_reused)
{
  check_reads(GetParam(), false, 256 * 1024);
}
TEST_P(AdaptiveTcpMssIntegration, small_responses_are_reused)
{
  check_reads(GetParam(), true, 16 * 1024);
}
TEST_P(AdaptiveTcpMssIntegration, only_small_peer_fails_open_and_stops_churning)
{
  check_reads(
    GetParam(), true, 256 * 1024, kvikio::detail::AdaptiveTcpMssPolicy::retirement_budget + 4);
}

// Synthetic resolver records are confined to the test easy handles. Production uses curl's
// normal resolution/cache unmodified; these records let us deterministically order peers.
class ResolvedEndpoint : public kvikio::HttpEndpoint {
 public:
  ResolvedEndpoint(unsigned short port, std::string const& addresses)
    : HttpEndpoint("http://mss.test:" + std::to_string(port) + "/data")
  {
    auto record = "mss.test:" + std::to_string(port) + ":" + addresses;
    _addresses  = curl_slist_append(nullptr, record.c_str());
    if (!_addresses) { throw std::bad_alloc{}; }
  }
  ~ResolvedEndpoint() override { curl_slist_free_all(_addresses); }
  void setopt(kvikio::CurlHandle& curl) override
  {
    HttpEndpoint::setopt(curl);
    curl.setopt(CURLOPT_RESOLVE, _addresses);
    curl.setopt(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl.setopt(CURLOPT_CONNECTTIMEOUT_MS, 1000L);
  }

 private:
  curl_slist* _addresses{};
};

void check_peer_selection(kvikio::RemoteIOBackend backend, bool alternative_alive)
{
  using kvikio::detail::adaptive_tcp_mss_policy;
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  auto const previous_backend  = kvikio::defaults::remote_io_backend();
  auto const previous_attempts = kvikio::defaults::http_max_attempts();
  kvikio::defaults::set_remote_io_backend(backend);
  kvikio::defaults::set_http_max_attempts(1);  // Fail-open must not depend on HTTP retries.
  constexpr std::size_t size = 256 * 1024;
  SmallMssServer small{size};
  std::unique_ptr<SmallMssServer> jumbo;
  if (alternative_alive) {
    jumbo = std::make_unique<SmallMssServer>(size, "127.0.0.2", 12000, small.port());
  }
  auto const origin = "http://mss.test:" + std::to_string(small.port());
  auto& policy      = adaptive_tcp_mss_policy();
  std::vector<char> buffer(size);
  if (jumbo) {
    // Obtain actual jumbo TCP evidence with a private easy handle, not a fabricated sample.
    kvikio::CurlHandle warm{
      kvikio::LibCurl::UniqueHandlePtr(curl_easy_init(), curl_easy_cleanup), __FILE__, "warm"};
    ResolvedEndpoint endpoint{small.port(), "127.0.0.2"};
    endpoint.setopt(warm);
    warm.setopt(CURLOPT_FORBID_REUSE, 1L);
    warm.setopt(
      CURLOPT_WRITEFUNCTION, +[](char*, size_t n, size_t size, void*) { return n * size; });
    warm.perform();
  } else {
    // Stale proof: the formerly jumbo alternative now refuses connections.
    policy.observe({origin, "127.0.0.1", AF_INET, size, 9000, 8948, "127.0.0.2"});
  }
  auto const before = policy.stats();
  {
    kvikio::RemoteHandle remote{
      std::make_unique<ResolvedEndpoint>(small.port(), "127.0.0.1,127.0.0.2"), size};
    for (int i = 0; i < 4; ++i) {
      std::fill(buffer.begin(), buffer.end(), '\0');
      EXPECT_EQ(remote.pread(buffer.data(), size, 0, size).get(), size);
      EXPECT_TRUE(std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == 'x'; }));
    }
  }
  bool const enabled = kvikio::defaults::remote_adaptive_tcp_mss();
  EXPECT_EQ(small.requests, enabled && jumbo ? 1 : 4);
  EXPECT_EQ(small.connections, enabled && !jumbo ? 2 : 1);
  if (jumbo) {
    EXPECT_EQ(jumbo->requests, enabled ? 4 : 1);  // Includes the warm-up; no replayed GETs.
    EXPECT_EQ(jumbo->connections, enabled ? 2 : 1);
    EXPECT_EQ(policy.stats().jumbo - before.jumbo, enabled ? 3 : 0);
  }
  EXPECT_EQ(policy.stats().retired - before.retired, enabled ? 1 : 0);
  EXPECT_EQ(policy.stats().fallback - before.fallback, enabled && !jumbo ? 3 : 0);
  kvikio::defaults::set_http_max_attempts(previous_attempts);
  kvikio::defaults::set_remote_io_backend(previous_backend);
}

TEST_P(AdaptiveTcpMssIntegration, skips_retired_peer_and_reuses_jumbo_alternative)
{
  check_peer_selection(GetParam(), true);
}

TEST_P(AdaptiveTcpMssIntegration, failed_alternative_falls_back_without_spending_http_retry)
{
  check_peer_selection(GetParam(), false);
}

TEST_P(AdaptiveTcpMssIntegration, all_resolved_peers_small_preserves_service)
{
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  auto const previous_backend  = kvikio::defaults::remote_io_backend();
  auto const previous_attempts = kvikio::defaults::http_max_attempts();
  kvikio::defaults::set_remote_io_backend(GetParam());
  kvikio::defaults::set_http_max_attempts(1);
  SmallMssServer first{256 * 1024};
  SmallMssServer second{256 * 1024, "127.0.0.2", 1024, first.port()};
  auto const origin = "http://mss.test:" + std::to_string(first.port());
  auto& policy      = kvikio::detail::adaptive_tcp_mss_policy();
  policy.observe({origin, "127.0.0.1", AF_INET, 256 * 1024, 9000, 8948, "127.0.0.9"});
  auto const before = policy.stats();
  kvikio::RemoteHandle remote{
    std::make_unique<ResolvedEndpoint>(first.port(), "127.0.0.1,127.0.0.2"), 256 * 1024};
  std::vector<char> buffer(256 * 1024);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(remote.pread(buffer.data(), buffer.size(), 0, buffer.size()).get(), buffer.size());
    EXPECT_TRUE(std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == 'x'; }));
  }
  EXPECT_EQ(first.requests + second.requests, 8);
  bool const enabled = kvikio::defaults::remote_adaptive_tcp_mss();
  EXPECT_EQ(first.connections + second.connections, enabled ? 3 : 1);
  EXPECT_EQ(policy.stats().retired - before.retired, enabled ? 2 : 0);
  EXPECT_EQ(policy.stats().fallback - before.fallback, enabled ? 6 : 0);
  kvikio::defaults::set_http_max_attempts(previous_attempts);
  kvikio::defaults::set_remote_io_backend(previous_backend);
}

TEST_P(AdaptiveTcpMssIntegration, unreachable_peers_still_fail_without_looping)
{
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  auto const previous_backend  = kvikio::defaults::remote_io_backend();
  auto const previous_attempts = kvikio::defaults::http_max_attempts();
  kvikio::defaults::set_remote_io_backend(GetParam());
  kvikio::defaults::set_http_max_attempts(1);
  SmallMssServer port_reservation{256 * 1024};
  auto const origin = "http://mss.test:" + std::to_string(port_reservation.port());
  auto& policy      = kvikio::detail::adaptive_tcp_mss_policy();
  policy.observe({origin, "127.0.0.1", AF_INET, 256 * 1024, 9000, 8948, "127.0.0.4"});
  policy.observe({origin, "127.0.0.1", AF_INET, 256 * 1024, 9000, 1412, "127.0.0.3"});
  kvikio::RemoteHandle remote{
    std::make_unique<ResolvedEndpoint>(port_reservation.port(), "127.0.0.3,127.0.0.4"), 256 * 1024};
  std::vector<char> buffer(256 * 1024);
  EXPECT_THROW(remote.pread(buffer.data(), buffer.size(), 0, buffer.size()).get(),
               std::runtime_error);
  if (kvikio::defaults::remote_adaptive_tcp_mss()) {
    EXPECT_FALSE(policy.avoid_peer(origin, AF_INET, "127.0.0.3"));
  }
  EXPECT_EQ(port_reservation.requests, 0);
  kvikio::defaults::set_http_max_attempts(previous_attempts);
  kvikio::defaults::set_remote_io_backend(previous_backend);
}

TEST_P(AdaptiveTcpMssIntegration, http_error_is_not_replayed_as_connection_fallback)
{
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  auto const previous_backend  = kvikio::defaults::remote_io_backend();
  auto const previous_attempts = kvikio::defaults::http_max_attempts();
  kvikio::defaults::set_remote_io_backend(GetParam());
  kvikio::defaults::set_http_max_attempts(1);
  SmallMssServer small{256 * 1024};
  SmallMssServer alternative{256 * 1024, "127.0.0.2", 12000, small.port()};
  small.status = alternative.status = 500;
  auto const origin                 = "http://mss.test:" + std::to_string(small.port());
  auto& policy                      = kvikio::detail::adaptive_tcp_mss_policy();
  policy.observe({origin, "127.0.0.1", AF_INET, 256 * 1024, 9000, 8948, "127.0.0.2"});
  policy.observe({origin, "127.0.0.1", AF_INET, 256 * 1024, 9000, 1412, "127.0.0.1"});
  kvikio::RemoteHandle remote{
    std::make_unique<ResolvedEndpoint>(small.port(), "127.0.0.1,127.0.0.2"), 256 * 1024};
  std::vector<char> buffer(256 * 1024);
  EXPECT_THROW(remote.pread(buffer.data(), buffer.size(), 0, buffer.size()).get(),
               std::runtime_error);
  EXPECT_TRUE(policy.avoid_peer(origin, AF_INET, "127.0.0.1"));
  EXPECT_EQ(small.requests, kvikio::defaults::remote_adaptive_tcp_mss() ? 0 : 1);
  EXPECT_EQ(alternative.requests, kvikio::defaults::remote_adaptive_tcp_mss() ? 1 : 0);
  kvikio::defaults::set_http_max_attempts(previous_attempts);
  kvikio::defaults::set_remote_io_backend(previous_backend);
}
TEST_P(AdaptiveTcpMssIntegration, partial_response_is_not_replayed_as_connection_fallback)
{
  kvikio::test::EnvVarContext env{{"NO_PROXY", "*"}, {"no_proxy", "*"}};
  kvikio::defaults::set_remote_io_backend(GetParam());
  kvikio::defaults::set_http_max_attempts(1);
  constexpr std::size_t size = 256 * 1024;
  SmallMssServer small{size};
  SmallMssServer alternative{size, "127.0.0.2", 12000, small.port()};
  small.truncate_body = alternative.truncate_body = true;
  auto const origin = "http://mss.test:" + std::to_string(small.port());
  auto& policy      = kvikio::detail::adaptive_tcp_mss_policy();
  policy.observe({origin, "127.0.0.1", AF_INET, size, 9000, 8948, "127.0.0.2"});
  policy.observe({origin, "127.0.0.1", AF_INET, size, 9000, 1412, "127.0.0.1"});
  auto const before = policy.stats();
  kvikio::RemoteHandle remote{
    std::make_unique<ResolvedEndpoint>(small.port(), "127.0.0.1,127.0.0.2"), size};
  std::vector<char> buffer(size);
  EXPECT_THROW(remote.pread(buffer.data(), size, 0, size).get(), std::runtime_error);
  EXPECT_TRUE(policy.avoid_peer(origin, AF_INET, "127.0.0.1"));
  EXPECT_EQ(small.requests, kvikio::defaults::remote_adaptive_tcp_mss() ? 0 : 1);
  EXPECT_EQ(alternative.requests, kvikio::defaults::remote_adaptive_tcp_mss() ? 1 : 0);
  EXPECT_EQ(policy.stats().retired, before.retired);
  EXPECT_EQ(policy.stats().fallback, before.fallback);
}

class ProxyEndpoint : public kvikio::HttpEndpoint {
 public:
  explicit ProxyEndpoint(unsigned short port)
    : HttpEndpoint("http://mss-proxy.test:" + std::to_string(port) + "/data"),
      proxy_{"http://127.0.0.1:" + std::to_string(port)}
  {
  }

  void setopt(kvikio::CurlHandle& curl) override
  {
    HttpEndpoint::setopt(curl);
    curl.setopt(CURLOPT_PROXY, proxy_.c_str());
    curl.setopt(CURLOPT_NOPROXY, "");
    curl.setopt(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl.setopt(CURLOPT_CONNECTTIMEOUT_MS, 1000L);
  }

 private:
  std::string proxy_;
};

TEST_P(AdaptiveTcpMssIntegration, proxy_is_not_filtered_or_retired)
{
  kvikio::defaults::set_remote_io_backend(GetParam());
  kvikio::defaults::set_http_max_attempts(1);
  constexpr std::size_t size = 256 * 1024;
  SmallMssServer proxy{size};
  auto const origin = "http://mss-proxy.test:" + std::to_string(proxy.port());
  auto& policy      = kvikio::detail::adaptive_tcp_mss_policy();
  // The proxy happens to use a previously retired peer address. Neither the open-socket
  // filter nor completed-response retirement may apply origin path evidence to a proxy.
  policy.observe({origin, "127.0.0.1", AF_INET, size, 9000, 8948, "127.0.0.2"});
  policy.observe({origin, "127.0.0.1", AF_INET, size, 9000, 1412, "127.0.0.1"});
  auto const before = policy.stats();
  kvikio::RemoteHandle remote{std::make_unique<ProxyEndpoint>(proxy.port()), size};
  std::vector<char> buffer(size);
  for (int i = 0; i < 2; ++i) {
    std::fill(buffer.begin(), buffer.end(), '\0');
    EXPECT_EQ(remote.pread(buffer.data(), size, 0, size).get(), size);
    EXPECT_TRUE(std::all_of(buffer.begin(), buffer.end(), [](char c) { return c == 'x'; }));
  }
  EXPECT_EQ(proxy.requests, 2);
  EXPECT_EQ(proxy.connections, 1);
  EXPECT_TRUE(policy.avoid_peer(origin, AF_INET, "127.0.0.1"));
  EXPECT_EQ(policy.stats().retired, before.retired);
  EXPECT_EQ(policy.stats().fallback, before.fallback);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         AdaptiveTcpMssIntegration,
                         testing::Values(kvikio::RemoteIOBackend::EASY_THREADPOOL,
                                         kvikio::RemoteIOBackend::MULTI_POLL));
}  // namespace
