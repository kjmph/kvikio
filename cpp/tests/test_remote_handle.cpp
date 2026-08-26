/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
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
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

#include "utils/env.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

class OneRequestHttpServer {
 public:
  OneRequestHttpServer()
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

  ~OneRequestHttpServer()
  {
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  [[nodiscard]] uint16_t port() const noexcept { return _port; }

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
    auto const client = ::accept(_listen_fd, nullptr, nullptr);
    if (client < 0) { return; }

    char buffer[4096];
    while (_request.find("\r\n\r\n") == std::string::npos) {
      auto const received = ::recv(client, buffer, sizeof(buffer), 0);
      if (received <= 0) { break; }
      _request.append(buffer, static_cast<std::size_t>(received));
    }

    constexpr char response[] = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    std::ignore               = send_all(client, response, sizeof(response) - 1);
    ::shutdown(client, SHUT_RDWR);
    ::close(client);
  }

  int _listen_fd = -1;
  uint16_t _port = 0;
  std::thread _thread;
  std::string _request;
};

}  // namespace

class CountingEndpoint : public kvikio::RemoteEndpoint {
 public:
  CountingEndpoint() : RemoteEndpoint{kvikio::RemoteEndpointType::HTTP} {}

  void setopt(kvikio::CurlHandle&) override { ++setopt_calls; }

  std::string str() const override { return "http://example.com/test"; }

  std::size_t get_file_size() override { return file_size; }

  void setup_range_request(kvikio::CurlHandle&, std::size_t, std::size_t) override
  {
    ++range_request_calls;
  }

  std::size_t file_size{100};
  int setopt_calls{};
  int range_request_calls{};
};

class RemoteHandleTest : public testing::Test {
 protected:
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
  OneRequestHttpServer server;
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
