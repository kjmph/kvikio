/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <curl/curl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/direct_receive_cuda.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>

#include "utils/env.hpp"
#include "utils/utils.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

class LocalHttpServer {
 public:
  explicit LocalHttpServer(std::string body                   = {},
                           bool exact_range                   = false,
                           std::size_t transient_failures     = 0,
                           std::size_t pause_after_body_bytes = 0)
    : _body{std::move(body)},
      _exact_range{exact_range},
      _transient_failures{transient_failures},
      _pause_after_body_bytes{pause_after_body_bytes}
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error(std::strerror(errno)); }

    int reuse = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(_listen_fd, 1) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }

    socklen_t address_length = sizeof(address);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &address_length) != 0) {
      auto const message = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      _listen_fd = -1;
      throw std::runtime_error(message);
    }
    _port   = ntohs(address.sin_port);
    _thread = std::thread{[this] { serve(); }};
  }

  ~LocalHttpServer()
  {
    resume_body();
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  [[nodiscard]] uint16_t port() const noexcept { return _port; }

  [[nodiscard]] bool wait_until_accepted(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_state_mutex};
    return _state_cv.wait_for(lock, timeout, [this] { return _accepted; });
  }

  [[nodiscard]] bool wait_until_body_paused(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock{_state_mutex};
    return _state_cv.wait_for(lock, timeout, [this] { return _body_paused; });
  }

  void resume_body() noexcept
  {
    try {
      {
        std::lock_guard lock{_state_mutex};
        _resume_body = true;
      }
      _state_cv.notify_all();
    } catch (...) {
    }
  }

  [[nodiscard]] std::string const& request()
  {
    if (_thread.joinable()) { _thread.join(); }
    return _request;
  }

 private:
  static bool send_all(int fd, char const* data, std::size_t size)
  {
    while (size != 0) {
      auto const sent = ::send(fd, data, size, MSG_NOSIGNAL);
      if (sent <= 0) { return false; }
      data += sent;
      size -= static_cast<std::size_t>(sent);
    }
    return true;
  }

  void serve()
  {
    for (std::size_t attempt = 0; attempt <= _transient_failures; ++attempt) {
      auto const client = ::accept(_listen_fd, nullptr, nullptr);
      if (client < 0) { return; }
      {
        std::lock_guard lock{_state_mutex};
        _accepted = true;
      }
      _state_cv.notify_all();

      std::string request;
      char buffer[4096];
      while (request.find("\r\n\r\n") == std::string::npos) {
        auto const received = ::recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0) { break; }
        request.append(buffer, static_cast<std::size_t>(received));
      }
      _request += request;

      auto response_header = std::string{};
      if (attempt < _transient_failures) {
        response_header =
          "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: "
          "close\r\n\r\n";
      } else if (_exact_range) {
        response_header = std::string{"HTTP/1.1 206 Partial Content\r\nContent-Length: "} +
                          std::to_string(_body.size()) + "\r\nContent-Range: bytes 0-" +
                          std::to_string(_body.size() - 1) + "/" + std::to_string(_body.size()) +
                          "\r\nConnection: close\r\n\r\n";
      } else {
        response_header = std::string{"HTTP/1.1 200 OK\r\nContent-Length: "} +
                          std::to_string(_body.size()) + "\r\nConnection: close\r\n\r\n";
      }
      if (!send_all(client, response_header.data(), response_header.size())) {
        ::close(client);
        return;
      }
      if (attempt >= _transient_failures) {
        auto const prefix = std::min(_pause_after_body_bytes, _body.size());
        if (!send_all(client, _body.data(), prefix)) {
          ::close(client);
          return;
        }
        if (prefix != 0 && prefix < _body.size()) {
          std::unique_lock lock{_state_mutex};
          _body_paused = true;
          _state_cv.notify_all();
          _state_cv.wait(lock, [this] { return _resume_body; });
        }
        std::ignore = send_all(client, _body.data() + prefix, _body.size() - prefix);
      }
      ::shutdown(client, SHUT_RDWR);
      ::close(client);
    }
  }

  int _listen_fd = -1;
  uint16_t _port = 0;
  std::thread _thread;
  std::string _request;
  std::string _body;
  bool _exact_range{};
  std::size_t _transient_failures{};
  std::size_t _pause_after_body_bytes{};
  std::mutex _state_mutex;
  std::condition_variable _state_cv;
  bool _accepted{};
  bool _body_paused{};
  bool _resume_body{};
};

template <typename Predicate>
[[nodiscard]] bool wait_until(Predicate&& predicate, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

std::atomic<std::size_t> stream_synchronize_calls{};
std::atomic<std::size_t> context_synchronize_calls{};
decltype(kvikio::cudaAPI::instance().StreamSynchronize) saved_stream_synchronize{};
decltype(kvikio::cudaAPI::instance().CtxSynchronize) saved_context_synchronize{};

CUresult CUDAAPI fail_stream_synchronize(CUstream)
{
  ++stream_synchronize_calls;
  return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI count_and_forward_context_synchronize()
{
  ++context_synchronize_calls;
  return saved_context_synchronize();
}

class ScopedCudaCompletionFailure {
 public:
  ScopedCudaCompletionFailure()
  {
    auto& stream_slot         = kvikio::cudaAPI::instance().StreamSynchronize;
    auto& context_slot        = kvikio::cudaAPI::instance().CtxSynchronize;
    saved_stream_synchronize  = stream_slot;
    saved_context_synchronize = context_slot;
    stream_synchronize_calls  = 0;
    context_synchronize_calls = 0;
    stream_slot               = &fail_stream_synchronize;
    context_slot              = &count_and_forward_context_synchronize;
  }

  ScopedCudaCompletionFailure(ScopedCudaCompletionFailure const&)            = delete;
  ScopedCudaCompletionFailure& operator=(ScopedCudaCompletionFailure const&) = delete;

  ~ScopedCudaCompletionFailure()
  {
    kvikio::cudaAPI::instance().StreamSynchronize = saved_stream_synchronize;
    kvikio::cudaAPI::instance().CtxSynchronize    = saved_context_synchronize;
  }
};

std::atomic<std::size_t> event_query_failures{};
decltype(kvikio::cudaAPI::instance().EventQuery) saved_event_query{};

CUresult CUDAAPI fail_one_event_query(CUevent event)
{
  if (event_query_failures.fetch_add(1, std::memory_order_relaxed) == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return saved_event_query(event);
}

class ScopedEventQueryFailure {
 public:
  ScopedEventQueryFailure()
  {
    auto& event_query = kvikio::cudaAPI::instance().EventQuery;
    saved_event_query = event_query;
    event_query_failures.store(0, std::memory_order_relaxed);
    event_query = &fail_one_event_query;
  }

  ScopedEventQueryFailure(ScopedEventQueryFailure const&)            = delete;
  ScopedEventQueryFailure& operator=(ScopedEventQueryFailure const&) = delete;

  ~ScopedEventQueryFailure() { kvikio::cudaAPI::instance().EventQuery = saved_event_query; }
};

std::atomic<bool> hold_event_queries{};
std::atomic<std::size_t> held_event_query_calls{};

CUresult CUDAAPI hold_event_query(CUevent event)
{
  if (hold_event_queries.load(std::memory_order_acquire)) {
    held_event_query_calls.fetch_add(1, std::memory_order_relaxed);
    return CUDA_ERROR_NOT_READY;
  }
  return saved_event_query(event);
}

class ScopedEventQueryGate {
 public:
  ScopedEventQueryGate()
  {
    auto& event_query = kvikio::cudaAPI::instance().EventQuery;
    saved_event_query = event_query;
    held_event_query_calls.store(0, std::memory_order_relaxed);
    hold_event_queries.store(true, std::memory_order_release);
    event_query = &hold_event_query;
  }

  ScopedEventQueryGate(ScopedEventQueryGate const&)            = delete;
  ScopedEventQueryGate& operator=(ScopedEventQueryGate const&) = delete;

  void release() noexcept { hold_event_queries.store(false, std::memory_order_release); }

  ~ScopedEventQueryGate()
  {
    release();
    kvikio::cudaAPI::instance().EventQuery = saved_event_query;
  }
};

std::atomic<std::size_t> event_record_calls{};
decltype(kvikio::cudaAPI::instance().EventRecord) saved_event_record{};

CUresult CUDAAPI fail_second_event_record(CUevent event, CUstream stream)
{
  auto const call = event_record_calls.fetch_add(1, std::memory_order_relaxed);
  if (call == 1) { return CUDA_ERROR_INVALID_VALUE; }
  return saved_event_record(event, stream);
}

class ScopedEventRecordFailure {
 public:
  ScopedEventRecordFailure()
  {
    auto& event_record = kvikio::cudaAPI::instance().EventRecord;
    saved_event_record = event_record;
    event_record_calls.store(0, std::memory_order_relaxed);
    event_record = &fail_second_event_record;
  }

  ScopedEventRecordFailure(ScopedEventRecordFailure const&)            = delete;
  ScopedEventRecordFailure& operator=(ScopedEventRecordFailure const&) = delete;

  ~ScopedEventRecordFailure() { kvikio::cudaAPI::instance().EventRecord = saved_event_record; }
};

}  // namespace

class CountingEndpoint : public kvikio::RemoteEndpoint {
 public:
  explicit CountingEndpoint(
    kvikio::RemoteEndpointType endpoint_type = kvikio::RemoteEndpointType::HTTP,
    std::string url                          = "http://example.com/test")
    : RemoteEndpoint{endpoint_type}, _url{std::move(url)}
  {
  }

  void setopt(kvikio::CurlHandle&) override { ++setopt_calls; }

  std::string str() const override { return _url; }

  [[nodiscard]] bool supports_exact_http_range() const noexcept override
  {
    return remote_endpoint_type() != kvikio::RemoteEndpointType::WEBHDFS &&
           remote_endpoint_type() != kvikio::RemoteEndpointType::AUTO;
  }

  [[nodiscard]] bool uses_origin_tls() const noexcept override
  {
    return _url.starts_with("https://");
  }

  std::size_t get_file_size() override { return file_size; }

  void setup_range_request(kvikio::CurlHandle&, std::size_t, std::size_t) override
  {
    ++range_request_calls;
  }

  std::size_t file_size{100};
  int setopt_calls{};
  int range_request_calls{};

 private:
  std::string _url;
};

class RestoreRemoteIoBackend {
 public:
  RestoreRemoteIoBackend() : _backend{kvikio::defaults::remote_io_backend()} {}

  ~RestoreRemoteIoBackend() { kvikio::defaults::set_remote_io_backend(_backend); }

 private:
  kvikio::RemoteIOBackend _backend;
};

class RestoreRemoteDirectReceiveMode {
 public:
  RestoreRemoteDirectReceiveMode() : _mode{kvikio::defaults::remote_direct_receive_mode()} {}
  ~RestoreRemoteDirectReceiveMode() { kvikio::defaults::set_remote_direct_receive_mode(_mode); }

 private:
  kvikio::RemoteDirectReceiveMode _mode;
};

class RestoreHttpRetryPolicy {
 public:
  RestoreHttpRetryPolicy()
    : _max_attempts{kvikio::defaults::http_max_attempts()},
      _status_codes{kvikio::defaults::http_status_codes()}
  {
  }

  ~RestoreHttpRetryPolicy()
  {
    kvikio::defaults::set_http_max_attempts(_max_attempts);
    kvikio::defaults::set_http_status_codes(std::move(_status_codes));
  }

 private:
  std::size_t _max_attempts;
  std::vector<int> _status_codes;
};

class RemoteHandleTest : public testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    static std::once_flag configured;
    std::call_once(configured, [] {
      auto const slot_size = kvikio::detail::DirectReceiveSlotPool::minimum_slot_size();
      kvikio::defaults::set_remote_direct_receive_slot_size(slot_size);
      kvikio::defaults::set_remote_direct_receive_max_pinned_bytes(slot_size);
    });
  }

  void SetUp() override
  {
    _sample_urls = {
      // Endpoint type: S3
      {"s3://bucket-name/object-key-name", kvikio::RemoteEndpointType::S3_PUBLIC},
      {"s3://bucket-name/object-key-name-dir/object-key-name-file",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3.region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3.region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://bucket-name.s3-region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},
      {"https://s3-region-code.amazonaws.com/bucket-name/object-key-name",
       kvikio::RemoteEndpointType::S3_PUBLIC},

      // Endpoint type: S3 presigned URL
      {"https://bucket-name.s3.region-code.amazonaws.com/"
       "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&X-Amz-Credential=cred&"
       "X-Amz-SignedHeaders=host",
       kvikio::RemoteEndpointType::S3_PRESIGNED_URL},

      // Endpoint type: WebHDFS
      {"https://host:1234/webhdfs/v1/data.bin", kvikio::RemoteEndpointType::WEBHDFS},
    };
  }

  void TearDown() override {}

  void test_helper(kvikio::RemoteEndpointType expected_endpoint_type,
                   std::function<bool(const std::string&)> url_validity_checker)
  {
    for (auto const& [url, endpoint_type] : _sample_urls) {
      if (endpoint_type == expected_endpoint_type) {
        // Given that the URL is the expected endpoint type

        // Test URL validity checker
        EXPECT_TRUE(url_validity_checker(url));

        // Test unified interface
        {
          // Here we pass the 1-byte argument to RemoteHandle::open. For all endpoints except
          // kvikio::RemoteEndpointType::S3 in AUTO mode, this prevents querying the file size and
          // sending requests to the server, thus allowing us to use dummy URLs for testing.
          // For kvikio::RemoteEndpointType::S3 with AUTO, RemoteHandle::open sends a HEAD request
          // as a connectivity check (and reuses that size when nbytes is not provided). It will
          // fail on the syntactically valid dummy URL, and kvikio::RemoteEndpointType::S3_PUBLIC
          // will then be used as the endpoint.
          auto remote_handle =
            kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1);
          EXPECT_EQ(remote_handle.remote_endpoint_type(), expected_endpoint_type);
        }

        // Test explicit endpoint type specification
        {
          EXPECT_NO_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, expected_endpoint_type, std::nullopt, 1);
          });
        }
      } else {
        // Given that the URL is NOT the expected endpoint type

        // Test URL validity checker
        EXPECT_FALSE(url_validity_checker(url));

        // Test explicit endpoint type specification
        {
          EXPECT_ANY_THROW({
            auto remote_handle =
              kvikio::RemoteHandle::open(url, expected_endpoint_type, std::nullopt, 1);
          });
        }
      }
    }
  }

  std::vector<std::pair<std::string, kvikio::RemoteEndpointType>> _sample_urls;
};

TEST_F(RemoteHandleTest, s3_endpoint_constructor)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"},
                                          {"AWS_ENDPOINT_URL", "https://my_aws_endpoint_url"}};
  std::string url        = "https://my_aws_endpoint_url/bucket_name/object_name";
  std::string aws_region = "my_aws_region";
  // Use the overload where the full url and the optional aws_region are specified.
  kvikio::S3Endpoint s1(url, aws_region);

  std::string bucket_name = "bucket_name";
  std::string object_name = "object_name";
  // Use the other overload where the bucket and object names are specified.
  kvikio::S3Endpoint s2(std::make_pair(bucket_name, object_name));

  EXPECT_EQ(s1.str(), s2.str());
}

TEST_F(RemoteHandleTest, s3_endpoint_forwards_explicit_session_token_for_non_sts_key)
{
  LocalHttpServer server;
  auto const url = "http://127.0.0.1:" + std::to_string(server.port()) + "/object";
  kvikio::S3Endpoint endpoint(
    url, "us-east-1", "CUSTOMACCESSKEY", "secret-access-key", "explicit-session-token");

  auto curl = create_curl_handle();
  endpoint.setopt(curl);
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_PROXY, "");
  curl.perform();

  auto const& request = server.request();
  std::string const expected{"x-amz-security-token: explicit-session-token\r\n"};
  auto const first = request.find(expected);
  ASSERT_NE(first, std::string::npos) << request;
  EXPECT_EQ(request.find(expected, first + expected.size()), std::string::npos) << request;
}

TEST_F(RemoteHandleTest, test_http_url)
{
  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{// Incorrect scheme
                                                "s3://example.com",
                                                "hdfs://example.com",
                                                // Missing file path
                                                "http://example.com"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::HttpEndpoint::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, http_endpoint_reports_direct_receive_transport_capabilities)
{
  kvikio::HttpEndpoint cleartext{"http://example.com/object"};
  EXPECT_TRUE(cleartext.supports_exact_http_range());
  EXPECT_FALSE(cleartext.uses_origin_tls());

  kvikio::HttpEndpoint tls{"HTTPS://example.com/object"};
  EXPECT_TRUE(tls.supports_exact_http_range());
  EXPECT_TRUE(tls.uses_origin_tls());
}

TEST_F(RemoteHandleTest, read_zero_size_returns_without_range_request)
{
  auto endpoint      = std::make_unique<CountingEndpoint>();
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);

  std::vector<char> output(1);
  EXPECT_EQ(remote_handle.read(output.data(), 0, 0), 0);
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
}

TEST_F(RemoteHandleTest, read_overflowing_range_throws_without_range_request)
{
  auto endpoint      = std::make_unique<CountingEndpoint>();
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);

  std::vector<char> output(1);
  auto constexpr file_offset = std::numeric_limits<std::size_t>::max() - 1;
  auto constexpr size        = std::size_t{4};
  EXPECT_THAT([&] { remote_handle.read(output.data(), size, file_offset); },
              ThrowsMessage<std::invalid_argument>(HasSubstr("cannot read ")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
}

TEST_F(RemoteHandleTest, device_read_failure_fences_context_before_rethrow)
{
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  LocalHttpServer server{"x"};
  auto endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(server.port()) + "/object");
  kvikio::RemoteHandle remote_handle(std::move(endpoint), 1);
  kvikio::test::DevBuffer<char> output(1);

  {
    ScopedCudaCompletionFailure const failure;
    EXPECT_THROW(std::ignore = remote_handle.read(output.ptr, 1), kvikio::CUfileException);
    EXPECT_EQ(stream_synchronize_calls.load(), 1);
    EXPECT_EQ(context_synchronize_calls.load(), 1);
  }
}

TEST_F(RemoteHandleTest, completion_future_failure_precedes_device_transfer_submission)
{
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);

  auto endpoint      = std::make_unique<CountingEndpoint>();
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  kvikio::test::DevBuffer<char> output(1);

  kvikio::detail::inject_io_completion_future_failure_for_testing();
  EXPECT_THROW(std::ignore = remote_handle.pread(output.ptr, 1, 0, 1), std::bad_alloc);

  // Transfer construction occurred, but the completion wrapper failed before the reactor-pool
  // handoff. No asynchronous work can retain the caller-owned destination.
  EXPECT_EQ(endpoint_ptr->setopt_calls, 1);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 1);
}

TEST_F(RemoteHandleTest, multi_poll_copied_stream_rotates_one_bounded_direct_receive_slot)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  auto const slot_size = kvikio::detail::DirectReceiveSlotPool::minimum_slot_size();
  kvikio::reset_remote_direct_receive_stats();

  std::string body(3 * slot_size + 137, '\0');
  for (std::size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<char>((i * 131U + 17U) & 0xffU);
  }
  LocalHttpServer server{body, true};
  auto endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(server.port()) + "/object");
  kvikio::RemoteHandle remote_handle(std::move(endpoint), body.size());
  kvikio::test::DevBuffer<char> output(body.size());

  auto completion = remote_handle.pread(output.ptr, body.size(), 0, body.size());
  EXPECT_EQ(completion.get(), body.size());
  std::string actual(body.size(), '\0');
  KVIKIO_CHECK_CUDA(cudaMemcpy(actual.data(), output.ptr, actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, body);

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 1);
  EXPECT_EQ(stats.strict_rx_transfers_activated, 0);
  EXPECT_EQ(stats.strict_rx_transfers_completed, 0);
  EXPECT_EQ(stats.transfers_fallback, 1);
  EXPECT_EQ(stats.fallback_ineligible_request, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.copied_stream_body_bytes, body.size());
  EXPECT_EQ(stats.copied_stream_h2d_bytes, body.size());
  EXPECT_GE(stats.copied_stream_raw_received_bytes, body.size());
  EXPECT_GE(stats.copied_stream_h2d_batches, 3);
  EXPECT_GE(stats.pinned_slots_acquired, 3);

  auto const pool = kvikio::detail::DirectReceiveSlotPool::instance().snapshot();
  EXPECT_EQ(pool.checked_out_slots, 0);
  EXPECT_EQ(pool.free_slots, 1);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, multi_poll_direct_receive_prioritizes_a_paused_read)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);

  auto const slot_size = kvikio::detail::DirectReceiveSlotPool::minimum_slot_size();
  std::string first_body(2 * slot_size + 137, 'a');
  std::string second_body(2 * slot_size + 251, 'b');
  LocalHttpServer first_server{first_body, true, 0, slot_size};
  LocalHttpServer second_server{second_body, true};
  auto first_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(first_server.port()) + "/first");
  auto second_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(second_server.port()) + "/second");
  kvikio::RemoteHandle first_handle(std::move(first_endpoint), first_body.size());
  kvikio::RemoteHandle second_handle(std::move(second_endpoint), second_body.size());
  kvikio::test::DevBuffer<char> first_output(first_body.size());
  kvikio::test::DevBuffer<char> second_output(second_body.size());

  kvikio::reset_remote_direct_receive_stats();
  ScopedEventQueryGate cuda_gate;
  auto first = first_handle.pread(first_output.ptr, first_body.size(), 0, first_body.size());
  EXPECT_TRUE(first_server.wait_until_body_paused(std::chrono::seconds{5}));
  EXPECT_TRUE(wait_until([] { return held_event_query_calls.load(std::memory_order_relaxed) != 0; },
                         std::chrono::seconds{5}));

  // The first read is paused without a receive slot while its first H2D remains gated. Queue a new
  // read, then let that slot recycle. A pending-first reactor would admit the second connection;
  // waiter priority must instead reinstall the slot on the already-paused first transfer.
  auto second = second_handle.pread(second_output.ptr, second_body.size(), 0, second_body.size());
  cuda_gate.release();
  EXPECT_TRUE(
    wait_until([] { return kvikio::remote_direct_receive_stats().pinned_slots_acquired >= 2; },
               std::chrono::seconds{5}));
  EXPECT_FALSE(second_server.wait_until_accepted(std::chrono::milliseconds{500}));

  first_server.resume_body();
  EXPECT_EQ(first.get(), first_body.size());
  EXPECT_EQ(second.get(), second_body.size());

  std::string first_actual(first_body.size(), '\0');
  std::string second_actual(second_body.size(), '\0');
  KVIKIO_CHECK_CUDA(
    cudaMemcpy(first_actual.data(), first_output.ptr, first_actual.size(), cudaMemcpyDeviceToHost));
  KVIKIO_CHECK_CUDA(cudaMemcpy(
    second_actual.data(), second_output.ptr, second_actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(first_actual, first_body);
  EXPECT_EQ(second_actual, second_body);

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 2);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 2);
  EXPECT_GE(stats.pinned_slots_acquired, 6);
  EXPECT_GT(stats.pinned_slot_exhaustions, 0);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, multi_poll_direct_receive_requeues_a_retryable_http_failure)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  RestoreHttpRetryPolicy const restore_retry;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::defaults::set_http_max_attempts(2);
  kvikio::defaults::set_http_status_codes({503});
  kvikio::reset_remote_direct_receive_stats();

  std::string body(257, 'r');
  LocalHttpServer server{body, true, 1};
  auto endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(server.port()) + "/object");
  kvikio::RemoteHandle remote_handle(std::move(endpoint), body.size());
  kvikio::test::DevBuffer<char> output(body.size());

  auto completion = remote_handle.pread(output.ptr, body.size(), 0, body.size());
  EXPECT_EQ(completion.get(), body.size());
  std::string actual(body.size(), '\0');
  KVIKIO_CHECK_CUDA(cudaMemcpy(actual.data(), output.ptr, actual.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, body);

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.transfers_requested, 1);
  EXPECT_EQ(stats.retries, 1);
  EXPECT_EQ(stats.copied_stream_transfers_completed, 1);
  EXPECT_EQ(stats.transfers_failed, 0);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, multi_poll_cuda_failure_is_not_published_as_network_success)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::reset_remote_direct_receive_stats();

  std::string body(257, 'x');
  LocalHttpServer failed_server{body, true};
  auto failed_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(failed_server.port()) + "/object");
  kvikio::RemoteHandle failed_handle(std::move(failed_endpoint), body.size());
  kvikio::test::DevBuffer<char> failed_output(body.size());

  {
    ScopedEventQueryFailure const failure;
    auto completion = failed_handle.pread(failed_output.ptr, body.size(), 0, body.size());
    EXPECT_THROW(std::ignore = completion.get(), kvikio::CUfileException);
    EXPECT_GE(event_query_failures.load(std::memory_order_relaxed), 1);
  }

  auto const failed_stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(failed_stats.transfers_requested, 1);
  EXPECT_EQ(failed_stats.copied_stream_transfers_completed, 0);
  EXPECT_EQ(failed_stats.transfers_failed, 1);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);

  // A safely fenced per-transfer CUDA error must not poison the process-lifetime reactor pool.
  LocalHttpServer recovery_server{body, true};
  auto recovery_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(recovery_server.port()) + "/object");
  kvikio::RemoteHandle recovery_handle(std::move(recovery_endpoint), body.size());
  kvikio::test::DevBuffer<char> recovery_output(body.size());
  auto recovery = recovery_handle.pread(recovery_output.ptr, body.size(), 0, body.size());
  EXPECT_EQ(recovery.get(), body.size());
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, multi_poll_post_enqueue_cuda_failure_remains_transfer_local)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::reset_remote_direct_receive_stats();

  std::string body(257, 'e');
  LocalHttpServer failed_server{body, true};
  auto failed_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(failed_server.port()) + "/object");
  kvikio::RemoteHandle failed_handle(std::move(failed_endpoint), body.size());
  kvikio::test::DevBuffer<char> failed_output(body.size());

  {
    ScopedEventRecordFailure const failure;
    auto completion = failed_handle.pread(failed_output.ptr, body.size(), 0, body.size());
    EXPECT_THROW(std::ignore = completion.get(), kvikio::CUfileException);
    EXPECT_GE(event_record_calls.load(std::memory_order_relaxed), 2);
  }
  EXPECT_EQ(kvikio::remote_direct_receive_stats().transfers_failed, 1);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);

  LocalHttpServer recovery_server{body, true};
  auto recovery_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(recovery_server.port()) + "/object");
  kvikio::RemoteHandle recovery_handle(std::move(recovery_endpoint), body.size());
  kvikio::test::DevBuffer<char> recovery_output(body.size());
  auto recovery = recovery_handle.pread(recovery_output.ptr, body.size(), 0, body.size());
  EXPECT_EQ(recovery.get(), body.size());
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, multi_poll_pre_enqueue_cuda_failure_remains_transfer_local)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
  kvikio::reset_remote_direct_receive_stats();

  std::string body(257, 'p');
  LocalHttpServer failed_server{body, true};
  auto failed_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(failed_server.port()) + "/object");
  kvikio::RemoteHandle failed_handle(std::move(failed_endpoint), body.size());
  kvikio::test::DevBuffer<char> failed_output(body.size());

  kvikio::detail::DirectReceiveCudaBatch::inject_pre_submit_allocation_failure_for_testing();
  auto completion = failed_handle.pread(failed_output.ptr, body.size(), 0, body.size());
  EXPECT_THROW(std::ignore = completion.get(), std::bad_alloc);
  EXPECT_EQ(kvikio::remote_direct_receive_stats().transfers_failed, 1);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);

  // The preparation error preceded every CUDA enqueue and must not poison the singleton reactor
  // pool or its bounded receive-slot pool.
  LocalHttpServer recovery_server{body, true};
  auto recovery_endpoint = std::make_unique<kvikio::HttpEndpoint>(
    "http://127.0.0.1:" + std::to_string(recovery_server.port()) + "/object");
  kvikio::RemoteHandle recovery_handle(std::move(recovery_endpoint), body.size());
  kvikio::test::DevBuffer<char> recovery_output(body.size());
  auto recovery = recovery_handle.pread(recovery_output.ptr, body.size(), 0, body.size());
  EXPECT_EQ(recovery.get(), body.size());
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, direct_receive_require_rejects_cleartext_before_publication)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::REQUIRE);

  auto endpoint      = std::make_unique<CountingEndpoint>();
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  kvikio::test::DevBuffer<char> output(1);

  EXPECT_THAT([&] { std::ignore = remote_handle.pread(output.ptr, 1, 0, 1); },
              ThrowsMessage<std::runtime_error>(HasSubstr("REQUIRE needs an HTTPS endpoint")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, direct_receive_require_rejects_webhdfs_before_publication)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  RestoreRemoteIoBackend const restore_backend;
  RestoreRemoteDirectReceiveMode const restore_mode;
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::REQUIRE);

  auto endpoint      = std::make_unique<CountingEndpoint>(kvikio::RemoteEndpointType::WEBHDFS,
                                                     "https://example.com/webhdfs/v1/test");
  auto* endpoint_ptr = endpoint.get();
  kvikio::RemoteHandle remote_handle(std::move(endpoint), endpoint_ptr->file_size);
  kvikio::test::DevBuffer<char> output(1);

  EXPECT_THAT(
    [&] { std::ignore = remote_handle.pread(output.ptr, 1, 0, 1); },
    ThrowsMessage<std::runtime_error>(HasSubstr("needs an exact-range HTTP or S3 endpoint")));
  EXPECT_EQ(endpoint_ptr->setopt_calls, 0);
  EXPECT_EQ(endpoint_ptr->range_request_calls, 0);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(RemoteHandleTest, test_s3_url)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_REGION", "my_aws_default_region"},
                                          {"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"}};

  {
    test_helper(kvikio::RemoteEndpointType::S3_PUBLIC, kvikio::S3Endpoint::is_url_valid);
  }

  // AWS_REGION is the standard SDK setting and takes precedence over the legacy
  // AWS_DEFAULT_REGION fallback when no explicit region argument is supplied.
  {
    kvikio::test::EnvVarContext region_env{{"AWS_REGION", "preferred_region"},
                                           {"AWS_DEFAULT_REGION", "fallback_region"}};
    EXPECT_EQ(kvikio::S3Endpoint::url_from_bucket_and_object(
                "bucket-name", "object-key-name", std::nullopt, std::nullopt),
              "https://bucket-name.s3.preferred_region.amazonaws.com/object-key-name");
  }

  // Public S3 accepts the object-store notation at its API boundary, but libcurl must receive an
  // HTTP(S) transport URL. Supplying nbytes avoids network access while still exposing the endpoint
  // selected by RemoteHandle::open().
  {
    auto remote_handle = kvikio::RemoteHandle::open("s3://bucket-name/path/to/object-key-name",
                                                    kvikio::RemoteEndpointType::S3_PUBLIC,
                                                    std::nullopt,
                                                    1);
    EXPECT_EQ(remote_handle.endpoint().str(),
              "https://bucket-name.s3.my_aws_default_region.amazonaws.com/path/to/object-key-name");
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{
      // Lack object-name
      "s3://bucket-name",
      "https://bucket-name.s3.region-code.amazonaws.com",
      // Presigned URL
      "https://bucket-name.s3.region-code.amazonaws.com/"
      "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&X-Amz-Credential="
      "cred&"
      "X-Amz-SignedHeaders=host"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::S3Endpoint::is_url_valid(invalid_url));
    }
  }

  // S3_PUBLIC is not in the allowlist. So when the connectivity check fails on the dummy URL,
  // KvikIO cannot fall back to S3_PUBLIC.
  {
    EXPECT_ANY_THROW({
      kvikio::RemoteHandle::open(
        "s3://bucket-name/object-key-name",
        kvikio::RemoteEndpointType::AUTO,
        std::vector<kvikio::RemoteEndpointType>{kvikio::RemoteEndpointType::S3,
                                                kvikio::RemoteEndpointType::HTTP},
        1);
    });
  }
}

TEST_F(RemoteHandleTest, test_s3_url_with_presigned_url)
{
  {
    test_helper(kvikio::RemoteEndpointType::S3_PRESIGNED_URL,
                kvikio::S3EndpointWithPresignedUrl::is_url_valid);
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{
      // Presigned URL should not use S3 scheme
      "s3://bucket-name/object-key-name",

      // Completely missing query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name",

      // Missing key parameters ("X-Amz-..."") in query
      "https://bucket-name.s3.region-code.amazonaws.com/object-key-name?k0=v0&k1=v2"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::S3EndpointWithPresignedUrl::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, test_webhdfs_url)
{
  {
    test_helper(kvikio::RemoteEndpointType::WEBHDFS, kvikio::WebHdfsEndpoint::is_url_valid);
  }

  // Invalid URLs
  {
    std::vector<std::string> const invalid_urls{// Missing file
                                                "https://host:1234/webhdfs/v1",
                                                "https://host:1234/webhdfs/v1/",

                                                // Missing WebHDFS identifier
                                                "https://host:1234/data.bin",

                                                // Missing port number
                                                "https://host/webhdfs/v1/data.bin"};
    for (auto const& invalid_url : invalid_urls) {
      EXPECT_FALSE(kvikio::WebHdfsEndpoint::is_url_valid(invalid_url));
    }
  }
}

TEST_F(RemoteHandleTest, test_open)
{
  // Missing scheme
  {
    std::vector<std::string> const urls{
      "example.com/path", "example.com:8080/path", "//example.com/path", "://example.com/path"};
    for (auto const& url : urls) {
      EXPECT_THROW(
        { kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1); },
        std::runtime_error);
    }
  }

  // Unsupported type
  {
    std::string const url{"unsupported://example.com/path"};
    EXPECT_THAT(
      [&] { kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::AUTO, std::nullopt, 1); },
      ThrowsMessage<std::runtime_error>(HasSubstr("Unsupported endpoint URL")));
  }

  // Specified URL not in the allowlist
  {
    std::string const url{"https://host:1234/webhdfs/v1/data.bin"};
    std::vector<std::vector<kvikio::RemoteEndpointType>> const wrong_allowlists{
      {},
      {kvikio::RemoteEndpointType::S3},
    };
    for (auto const& wrong_allowlist : wrong_allowlists) {
      EXPECT_THAT(
        [&] {
          kvikio::RemoteHandle::open(url, kvikio::RemoteEndpointType::WEBHDFS, wrong_allowlist, 1);
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("is not in the allowlist")));
    }
  }

  // Invalid URLs
  {
    std::vector<std::pair<std::string, kvikio::RemoteEndpointType>> const invalid_urls{
      {"s3://bucket-name", kvikio::RemoteEndpointType::S3},
      {"https://bucket-name.s3.region-code.amazonaws.com/object-key-name",
       kvikio::RemoteEndpointType::S3_PRESIGNED_URL},
      {"https://host:1234/webhdfs/v1", kvikio::RemoteEndpointType::WEBHDFS},
      {"http://example.com", kvikio::RemoteEndpointType::HTTP},
    };
    for (auto const& [invalid_url, endpoint_type] : invalid_urls) {
      EXPECT_THAT([&] { kvikio::RemoteHandle::open(invalid_url, endpoint_type, std::nullopt, 1); },
                  ThrowsMessage<std::runtime_error>(HasSubstr("Invalid URL")));
    }
  }
}

TEST_F(RemoteHandleTest, test_infer_remote_endpoint_type)
{
  kvikio::test::EnvVarContext env_var_ctx{{"AWS_DEFAULT_REGION", "my_aws_default_region"},
                                          {"AWS_ACCESS_KEY_ID", "my_aws_access_key_id"},
                                          {"AWS_SECRET_ACCESS_KEY", "my_aws_secrete_access_key"}};

  EXPECT_EQ(kvikio::infer_remote_endpoint_type("s3://bucket-name/object-key-name"),
            kvikio::RemoteEndpointType::S3);
  EXPECT_EQ(kvikio::infer_remote_endpoint_type("https://host:1234/webhdfs/v1/data.bin"),
            kvikio::RemoteEndpointType::WEBHDFS);
  EXPECT_EQ(kvikio::infer_remote_endpoint_type("https://example.com/path/file.bin"),
            kvikio::RemoteEndpointType::HTTP);
  EXPECT_EQ(kvikio::infer_remote_endpoint_type(
              "https://bucket-name.s3.region-code.amazonaws.com/"
              "object-key-name?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=sig&"
              "X-Amz-Credential=cred&X-Amz-SignedHeaders=host"),
            kvikio::RemoteEndpointType::S3_PRESIGNED_URL);

  EXPECT_THAT([&] { kvikio::infer_remote_endpoint_type("unsupported://example.com/path"); },
              ThrowsMessage<std::runtime_error>(HasSubstr("Unsupported endpoint URL")));
  EXPECT_THAT([&] { kvikio::infer_remote_endpoint_type("example.com/path"); },
              ThrowsMessage<std::runtime_error>(HasSubstr("Bad scheme")));
}
