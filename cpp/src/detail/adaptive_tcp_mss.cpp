/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <kvikio/detail/adaptive_tcp_mss.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>

#include <curl/options.h>

#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>

#if defined(__linux__)
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

namespace kvikio::detail {

TcpMssDecision AdaptiveTcpMssPolicy::observe(TcpMssSample const& sample, Clock::time_point now)
{
  std::lock_guard const lock(_mutex);
  if (sample.body_bytes < minimum_body_bytes || sample.origin.empty() ||
      sample.local_address.empty() || sample.family == 0 || sample.path_mtu == 0 ||
      sample.receive_mss == 0) {
    ++_stats.unobservable;
    return TcpMssDecision::UNOBSERVABLE;
  }

  bool const jumbo = sample.path_mtu >= 8000 && sample.receive_mss >= 8000;
  bool const small = sample.path_mtu <= 2048 || sample.receive_mss <= 2048;
  Key const key{sample.origin, sample.local_address, sample.family};
  auto entry = _evidence.find(key);
  if (entry != _evidence.end() && now >= entry->second.expires) {
    _evidence.erase(entry);
    entry = _evidence.end();
  }

  if (jumbo) {
    if (entry == _evidence.end()) {
      // Do not evict live evidence: doing so could reset another origin's retirement budget.
      if (_evidence.size() >= maximum_origins) {
        std::erase_if(_evidence, [now](auto const& item) { return now >= item.second.expires; });
      }
      if (_evidence.size() < maximum_origins) {
        entry = _evidence.emplace(key, Evidence{now + evidence_lifetime}).first;
      }
    }
    if (entry != _evidence.end()) { entry->second.peers.erase(sample.peer_address); }
    ++_stats.jumbo;
    return TcpMssDecision::JUMBO;
  }
  if (entry == _evidence.end()) {
    ++_stats.unproven;
    return TcpMssDecision::UNPROVEN;
  }
  if (!small) {
    ++_stats.intermediate;
    return TcpMssDecision::INTERMEDIATE;
  }
  if (entry->second.suspended || entry->second.remaining == 0) {
    ++_stats.fallback;
    return TcpMssDecision::FALLBACK;
  }
  if (!sample.peer_address.empty()) { entry->second.peers.insert(sample.peer_address); }
  --entry->second.remaining;
  ++_stats.retired;
  return TcpMssDecision::RETIRE;
}

TcpMssStats AdaptiveTcpMssPolicy::stats() const
{
  std::lock_guard const lock(_mutex);
  return _stats;
}

void AdaptiveTcpMssPolicy::record_unobservable()
{
  std::lock_guard const lock(_mutex);
  ++_stats.unobservable;
}

bool AdaptiveTcpMssPolicy::avoid_peer(std::string const& origin,
                                      int family,
                                      std::string const& peer,
                                      Clock::time_point now) const
{
  std::lock_guard const lock(_mutex);
  Evidence const* evidence = nullptr;
  for (auto const& [key, value] : _evidence) {
    if (std::get<0>(key) != origin || std::get<2>(key) != family || now >= value.expires) {
      continue;
    }
    // A source interface cannot be determined here. Do not mix evidence from different paths.
    if (evidence != nullptr || value.suspended) { return false; }
    evidence = &value;
  }
  return evidence != nullptr && evidence->peers.contains(peer);
}

void AdaptiveTcpMssPolicy::allow_fallback(std::string const& origin, Clock::time_point now)
{
  std::lock_guard const lock(_mutex);
  for (auto& [key, evidence] : _evidence) {
    if (std::get<0>(key) == origin && now < evidence.expires) {
      evidence.suspended = true;
      evidence.peers.clear();
    }
  }
}

AdaptiveTcpMssPolicy& adaptive_tcp_mss_policy()
{
  // Reactors intentionally live until process exit; callbacks must have the same lifetime.
  static auto* policy = new AdaptiveTcpMssPolicy;
  return *policy;
}

namespace {
#if defined(__linux__) && defined(CURL_CONN_REUSE_RETIRE)
std::string effective_origin(CURL* easy)
{
  char* url = nullptr;
  if (curl_easy_getinfo(easy, CURLINFO_EFFECTIVE_URL, &url) != CURLE_OK || !url) { return {}; }
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> parsed{curl_url(), curl_url_cleanup};
  if (!parsed || curl_url_set(parsed.get(), CURLUPART_URL, url, 0) != CURLUE_OK) { return {}; }
  auto component = [&](CURLUPart part) -> std::string {
    char* raw         = nullptr;
    auto const result = curl_url_get(parsed.get(), part, &raw, CURLU_DEFAULT_PORT);
    std::unique_ptr<char, decltype(&curl_free)> owned{raw, curl_free};
    return result == CURLUE_OK && raw ? std::string{raw} : std::string{};
  };
  auto scheme = component(CURLUPART_SCHEME);
  auto host   = component(CURLUPART_HOST);
  auto port   = component(CURLUPART_PORT);
  if ((scheme != "https" && scheme != "http") || host.empty() || port.empty()) { return {}; }
  std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : static_cast<char>(c);
  });
  return scheme + "://" + host + ":" + port;
}

std::string numeric_address(sockaddr const* address)
{
  void const* raw;
  if (address->sa_family == AF_INET) {
    raw = &reinterpret_cast<sockaddr_in const*>(address)->sin_addr;
  } else if (address->sa_family == AF_INET6) {
    raw = &reinterpret_cast<sockaddr_in6 const*>(address)->sin6_addr;
  } else {
    return {};
  }
  char ip[INET6_ADDRSTRLEN];
  if (!inet_ntop(address->sa_family, raw, ip, sizeof(ip))) { return {}; }
  std::string result{ip};
  if (address->sa_family == AF_INET6) {
    result += "%" + std::to_string(reinterpret_cast<sockaddr_in6 const*>(address)->sin6_scope_id);
  }
  return result;
}

std::optional<TcpMssSample> sample_connection(CURL* easy, curl_socket_t socket)
{
  long proxy            = 0;
  long version          = 0;
  long status           = 0;
  curl_off_t body_bytes = 0;
  char* method          = nullptr;
  if (socket == CURL_SOCKET_BAD) { return std::nullopt; }
  if (curl_easy_getinfo(easy, CURLINFO_USED_PROXY, &proxy) != CURLE_OK || proxy) {
    return std::nullopt;
  }
  if (curl_easy_getinfo(easy, CURLINFO_HTTP_VERSION, &version) != CURLE_OK ||
      version != CURL_HTTP_VERSION_1_1) {
    return std::nullopt;
  }
  if (curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status) != CURLE_OK || status < 200 ||
      status >= 300) {
    return std::nullopt;
  }
  if (curl_easy_getinfo(easy, CURLINFO_EFFECTIVE_METHOD, &method) != CURLE_OK || !method ||
      std::strcmp(method, "GET") != 0) {
    return std::nullopt;
  }
  if (curl_easy_getinfo(easy, CURLINFO_SIZE_DOWNLOAD_T, &body_bytes) != CURLE_OK ||
      body_bytes < static_cast<curl_off_t>(AdaptiveTcpMssPolicy::minimum_body_bytes)) {
    return std::nullopt;
  }

  tcp_info info{};
  socklen_t info_length = sizeof(info);
  if (getsockopt(socket, IPPROTO_TCP, TCP_INFO, &info, &info_length) != 0 ||
      info_length < offsetof(tcp_info, tcpi_pmtu) + sizeof(info.tcpi_pmtu)) {
    return std::nullopt;
  }
  sockaddr_storage address{};
  socklen_t address_length = sizeof(address);
  if (getsockname(socket, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) {
    return std::nullopt;
  }
  auto local_key = numeric_address(reinterpret_cast<sockaddr*>(&address));
  if (local_key.empty()) { return std::nullopt; }

  auto origin = effective_origin(easy);
  if (origin.empty()) { return std::nullopt; }
  sockaddr_storage peer{};
  socklen_t peer_length = sizeof(peer);
  std::string peer_key;
  if (getpeername(socket, reinterpret_cast<sockaddr*>(&peer), &peer_length) == 0) {
    peer_key = numeric_address(reinterpret_cast<sockaddr*>(&peer));
  }
  return TcpMssSample{std::move(origin),
                      std::move(local_key),
                      address.ss_family,
                      static_cast<std::uint64_t>(body_bytes),
                      info.tcpi_pmtu,
                      info.tcpi_rcv_mss,
                      std::move(peer_key)};
}

int connection_reuse(CURL* easy, curl_socket_t socket, void*) noexcept
{
  // Never throw across curl's C callback boundary, or turn successful data into a retry.
  try {
    auto& policy = adaptive_tcp_mss_policy();
    auto sample  = sample_connection(easy, socket);
    if (!sample) {
      policy.record_unobservable();
      return CURL_CONN_REUSE_KEEP;
    }
    auto const decision = policy.observe(*sample);
    if (decision == TcpMssDecision::RETIRE) {
      KVIKIO_LOG_DEBUG(
        "Adaptive TCP MSS: retire connection after successful GET "
        "(PMTU={}, receive MSS={})",
        sample->path_mtu,
        sample->receive_mss);
      return CURL_CONN_REUSE_RETIRE;
    }
  } catch (...) {
    // Allocation/inspection failures are not transfer failures. Preserve normal connection reuse.
  }
  return CURL_CONN_REUSE_KEEP;
}
#endif
}  // namespace

void set_up_adaptive_tcp_mss(CURL* easy)
{
#if defined(__linux__) && defined(CURL_CONN_REUSE_RETIRE)
  // Experimental option numbers are not a runtime capability check. A different libcurl may
  // assign this number to an unrelated callback; validate its name and type before setting it.
  static bool const supported = [] {
    auto const* option = curl_easy_option_by_name("CONN_REUSEFUNCTION");
    return option != nullptr && option->id == CURLOPT_CONN_REUSEFUNCTION &&
           option->type == CURLOT_FUNCTION;
  }();
  KVIKIO_EXPECT(supported,
                "KVIKIO_REMOTE_ADAPTIVE_TCP_MSS requires a matching libcurl connection-reuse API",
                std::runtime_error);
  auto const result = curl_easy_setopt(easy, CURLOPT_CONN_REUSEFUNCTION, connection_reuse);
  KVIKIO_EXPECT(result == CURLE_OK,
                "KVIKIO_REMOTE_ADAPTIVE_TCP_MSS requires libcurl's connection-reuse callback",
                std::runtime_error);
#else
  (void)easy;
  KVIKIO_FAIL(
    "KVIKIO_REMOTE_ADAPTIVE_TCP_MSS requires Linux and libcurl's connection-reuse callback",
    std::runtime_error);
#endif
}

AdaptiveTcpMssConnection::AdaptiveTcpMssConnection(CURL* easy) : _easy{easy}
{
  set_up_adaptive_tcp_mss(easy);
#if defined(__linux__) && defined(CURL_CONN_REUSE_RETIRE)
  auto result = curl_easy_setopt(easy, CURLOPT_OPENSOCKETDATA, this);
  if (result == CURLE_OK) {
    result = curl_easy_setopt(easy, CURLOPT_OPENSOCKETFUNCTION, open_socket);
  }
  if (result != CURLE_OK) {
    (void)curl_easy_setopt(easy, CURLOPT_OPENSOCKETDATA, nullptr);
    KVIKIO_FAIL("Cannot install adaptive TCP MSS socket callback", std::runtime_error);
  }
#endif
}

AdaptiveTcpMssConnection::~AdaptiveTcpMssConnection() noexcept
{
  (void)curl_easy_setopt(_easy, CURLOPT_OPENSOCKETFUNCTION, nullptr);
  (void)curl_easy_setopt(_easy, CURLOPT_OPENSOCKETDATA, nullptr);
}

curl_socket_t AdaptiveTcpMssConnection::open_socket(void* data,
                                                    curlsocktype purpose,
                                                    curl_sockaddr* address) noexcept
{
#if defined(__linux__) && defined(CURL_CONN_REUSE_RETIRE)
  auto& self = *static_cast<AdaptiveTcpMssConnection*>(data);
  try {
    long proxy = 0, request_bytes = 0;
    char* method = nullptr;
    // Exclude proxies and any transfer that has already issued a request (redirects included).
    if (!self._unfiltered && purpose == CURLSOCKTYPE_IPCXN &&
        (address->socktype & ~(SOCK_CLOEXEC | SOCK_NONBLOCK)) == SOCK_STREAM &&
        curl_easy_getinfo(self._easy, CURLINFO_USED_PROXY, &proxy) == CURLE_OK && !proxy &&
        curl_easy_getinfo(self._easy, CURLINFO_REQUEST_SIZE, &request_bytes) == CURLE_OK &&
        request_bytes == 0 &&
        curl_easy_getinfo(self._easy, CURLINFO_EFFECTIVE_METHOD, &method) == CURLE_OK && method &&
        std::strcmp(method, "GET") == 0) {
      auto origin = effective_origin(self._easy);
      auto peer   = numeric_address(&address->addr);
      if (adaptive_tcp_mss_policy().avoid_peer(origin, address->family, peer)) {
        self._skipped_origin = std::move(origin);
        errno                = EHOSTUNREACH;
        return CURL_SOCKET_BAD;  // curl tries the next address in its existing resolver result.
      }
    }
  } catch (...) {
    // Inspection/allocation failures must preserve ordinary connection establishment.
  }
  return ::socket(address->family, address->socktype | SOCK_CLOEXEC, address->protocol);
#else
  return CURL_SOCKET_BAD;  // Construction rejects unsupported platforms before installing this.
#endif
}

bool AdaptiveTcpMssConnection::retry_unfiltered(CURLcode result) noexcept
{
#if defined(__linux__) && defined(CURL_CONN_REUSE_RETIRE)
  if (_unfiltered || _skipped_origin.empty()) { return false; }
  // A skipped working peer must remain usable if alternatives cannot establish a connection.
  // Never replay an HTTP request, an error response or a partially delivered response here.
  if (result != CURLE_COULDNT_CONNECT && result != CURLE_OPERATION_TIMEDOUT &&
      result != CURLE_SSL_CONNECT_ERROR && result != CURLE_PEER_FAILED_VERIFICATION) {
    _skipped_origin.clear();
    return false;
  }
  long request_bytes = -1, status = -1;
  curl_off_t body = -1;
  if (curl_easy_getinfo(_easy, CURLINFO_REQUEST_SIZE, &request_bytes) != CURLE_OK ||
      curl_easy_getinfo(_easy, CURLINFO_RESPONSE_CODE, &status) != CURLE_OK ||
      curl_easy_getinfo(_easy, CURLINFO_SIZE_DOWNLOAD_T, &body) != CURLE_OK || request_bytes != 0 ||
      status != 0 || body != 0) {
    _skipped_origin.clear();
    return false;
  }
  _unfiltered = true;
  try {
    adaptive_tcp_mss_policy().allow_fallback(_skipped_origin);
  } catch (...) {
  }
  return true;
#else
  return false;
#endif
}

}  // namespace kvikio::detail
