/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <kvikio/aws_credential_provider.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/remote_handle.hpp>

namespace {

using namespace std::chrono_literals;

class ScopedEnvironment {
 public:
  ~ScopedEnvironment()
  {
    for (auto it = _saved.rbegin(); it != _saved.rend(); ++it) {
      if (it->value.has_value()) {
        (void)::setenv(it->name.c_str(), it->value->c_str(), 1);
      } else {
        (void)::unsetenv(it->name.c_str());
      }
    }
  }

  void set(std::string name, std::string value)
  {
    save_once(name);
    if (::setenv(name.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error{std::strerror(errno)};
    }
  }

  void unset(std::string name)
  {
    save_once(name);
    if (::unsetenv(name.c_str()) != 0) { throw std::runtime_error{std::strerror(errno)}; }
  }

 private:
  struct SavedValue {
    std::string name;
    std::optional<std::string> value;
  };

  void save_once(std::string const& name)
  {
    for (auto const& saved : _saved) {
      if (saved.name == name) { return; }
    }
    auto const* current = std::getenv(name.c_str());
    _saved.push_back(
      {name, current == nullptr ? std::nullopt : std::optional<std::string>{current}});
  }

  std::vector<SavedValue> _saved;
};

struct ExpectedRequest {
  std::string method;
  std::string path;
  std::vector<std::string> required_headers;
  int status{200};
  std::string body;
  std::vector<std::string> response_headers;
  std::optional<std::size_t> content_length;
};

class ScriptedHttpServer {
 public:
  explicit ScriptedHttpServer(std::vector<ExpectedRequest> requests)
    : _expected{std::move(requests)}
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error{std::strerror(errno)}; }
    int reuse = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
      auto const error = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      throw std::runtime_error{error};
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(_listen_fd, 16) != 0) {
      auto const error = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      throw std::runtime_error{error};
    }
    socklen_t size = sizeof(address);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
      auto const error = std::string{std::strerror(errno)};
      ::close(_listen_fd);
      throw std::runtime_error{error};
    }
    _port   = ntohs(address.sin_port);
    _thread = std::thread{[this] { serve(); }};
  }

  ~ScriptedHttpServer() { stop(); }

  [[nodiscard]] std::string origin() const { return "http://127.0.0.1:" + std::to_string(_port); }

  [[nodiscard]] std::string finish(std::chrono::seconds timeout = 10s)
  {
    {
      std::unique_lock lock{_mutex};
      if (!_condition.wait_for(lock, timeout, [this] { return _done; })) {
        _failure += "timed out waiting for scripted requests; ";
      }
    }
    stop();
    std::lock_guard lock{_mutex};
    return _failure;
  }

  [[nodiscard]] std::size_t request_count() const
  {
    std::lock_guard lock{_mutex};
    return _requests.size();
  }

 private:
  static bool send_all(int fd, std::string const& value)
  {
    auto data = value.data();
    auto left = value.size();
    while (left != 0) {
      auto const written = ::send(fd, data, left, MSG_NOSIGNAL);
      if (written <= 0) { return false; }
      data += written;
      left -= static_cast<std::size_t>(written);
    }
    return true;
  }

  void record_failure(std::string failure)
  {
    std::lock_guard lock{_mutex};
    _failure += std::move(failure);
    _failure += "; ";
  }

  void serve()
  {
    for (auto const& expected : _expected) {
      auto const client = ::accept(_listen_fd, nullptr, nullptr);
      if (client < 0) { break; }

      std::string request;
      std::array<char, 4096> buffer{};
      while (request.find("\r\n\r\n") == std::string::npos) {
        auto const received = ::recv(client, buffer.data(), buffer.size(), 0);
        if (received <= 0) { break; }
        request.append(buffer.data(), static_cast<std::size_t>(received));
        if (request.size() > 64 * 1024) { break; }
      }
      {
        std::lock_guard lock{_mutex};
        _requests.push_back(request);
      }

      auto const request_line = expected.method + " " + expected.path + " HTTP/";
      if (!request.starts_with(request_line)) {
        record_failure("unexpected request line: " + request.substr(0, request.find("\r\n")));
      }
      for (auto const& header : expected.required_headers) {
        if (request.find(header) == std::string::npos) {
          record_failure("missing request header: " + header);
        }
      }

      std::string reason = expected.status == 200   ? "OK"
                           : expected.status == 206 ? "Partial Content"
                           : expected.status == 500 ? "Internal Server Error"
                                                    : "Error";
      std::ostringstream response;
      response << "HTTP/1.1 " << expected.status << ' ' << reason << "\r\n"
               << "Content-Length: " << expected.content_length.value_or(expected.body.size())
               << "\r\n";
      for (auto const& header : expected.response_headers) {
        response << header << "\r\n";
      }
      response << "Connection: close\r\n\r\n" << expected.body;
      if (!send_all(client, response.str())) { record_failure("failed to send response"); }
      ::shutdown(client, SHUT_RDWR);
      ::close(client);
    }
    {
      std::lock_guard lock{_mutex};
      _done = true;
    }
    _condition.notify_all();
  }

  void stop()
  {
    auto const fd = std::exchange(_listen_fd, -1);
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  std::vector<ExpectedRequest> _expected;
  int _listen_fd{-1};
  uint16_t _port{};
  std::thread _thread;
  mutable std::mutex _mutex;
  std::condition_variable _condition;
  bool _done{};
  std::string _failure;
  std::vector<std::string> _requests;
};

std::string format_expiration(std::chrono::system_clock::time_point expiration)
{
  auto const timestamp = std::chrono::system_clock::to_time_t(expiration);
  std::tm utc{};
  if (::gmtime_r(&timestamp, &utc) == nullptr) { throw std::runtime_error{"gmtime_r failed"}; }
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

std::string credential_document(std::string access_key,
                                std::string secret_key,
                                std::string token,
                                std::chrono::system_clock::time_point expiration,
                                std::string code = "Success")
{
  std::ostringstream json;
  json << "{\"Code\":\"" << code << "\","
       << "\"LastUpdated\":\"2026-08-30T00:00:00Z\","
       << "\"Type\":\"AWS-HMAC\","
       << "\"AccessKeyId\":\"" << access_key << "\","
       << "\"SecretAccessKey\":\"" << secret_key << "\","
       << "\"Token\":\"" << token << "\","
       << "\"Expiration\":\"" << format_expiration(expiration) << "\"}";
  return json.str();
}

std::vector<ExpectedRequest> imds_success(std::string access_key,
                                          std::string secret_key,
                                          std::string session_token,
                                          std::chrono::system_clock::time_point expiration,
                                          std::string metadata_token = "metadata-token",
                                          std::string role_name      = "test-role")
{
  return {{"PUT",
           "/latest/api/token",
           {"X-aws-ec2-metadata-token-ttl-seconds: 21600\r\n"},
           200,
           metadata_token},
          {"GET",
           "/latest/meta-data/iam/security-credentials/",
           {"X-aws-ec2-metadata-token: " + metadata_token + "\r\n"},
           200,
           role_name + "\n"},
          {"GET",
           "/latest/meta-data/iam/security-credentials/" + role_name,
           {"X-aws-ec2-metadata-token: " + metadata_token + "\r\n"},
           200,
           credential_document(
             std::move(access_key), std::move(secret_key), std::move(session_token), expiration)}};
}

void append_requests(std::vector<ExpectedRequest>& destination, std::vector<ExpectedRequest> source)
{
  destination.insert(destination.end(),
                     std::make_move_iterator(source.begin()),
                     std::make_move_iterator(source.end()));
}

class CredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    _http_attempts = kvikio::defaults::http_max_attempts();
    kvikio::defaults::set_http_max_attempts(1);
  }

  void TearDown() override { kvikio::defaults::set_http_max_attempts(_http_attempts); }

 private:
  std::size_t _http_attempts{};
};

TEST_F(CredentialProviderTest, static_credentials_are_validated_and_formatted)
{
  auto provider =
    kvikio::make_static_aws_credential_provider("ASIAEXPLICIT", "secret", "session-token");
  auto material = provider->get_auth_material();
  EXPECT_EQ(material->userpwd, "ASIAEXPLICIT:secret");
  ASSERT_TRUE(material->session_token_header.has_value());
  EXPECT_EQ(*material->session_token_header, "x-amz-security-token: session-token");

  EXPECT_THROW(kvikio::make_static_aws_credential_provider("ASIAEXPLICIT", "secret"),
               std::invalid_argument);
  EXPECT_THROW(kvikio::make_static_aws_credential_provider("access:key", "secret"),
               std::invalid_argument);
  EXPECT_THROW(kvikio::make_static_aws_credential_provider("access", "secret\nvalue"),
               std::invalid_argument);
  EXPECT_THROW(
    kvikio::make_static_aws_credential_provider("access", std::string{"secret\0suffix", 13}),
    std::invalid_argument);
}

TEST_F(CredentialProviderTest, legacy_braced_optional_constructor_remains_unambiguous)
{
  ScopedEnvironment environment;
  environment.set("AWS_ACCESS_KEY_ID", "AKIALEGACY");
  environment.set("AWS_SECRET_ACCESS_KEY", "legacy-secret");
  kvikio::S3Endpoint endpoint{"https://bucket.s3.us-east-1.amazonaws.com/object", "us-east-1", {}};
  EXPECT_EQ(endpoint.str(), "https://bucket.s3.us-east-1.amazonaws.com/object");
}

TEST_F(CredentialProviderTest, default_provider_is_process_shared_without_an_override)
{
  EXPECT_EQ(kvikio::make_default_aws_credential_provider(),
            kvikio::make_default_aws_credential_provider());
  EXPECT_NE(kvikio::make_default_aws_credential_provider("http://127.0.0.1:1"),
            kvikio::make_default_aws_credential_provider("http://127.0.0.1:1"));
}

TEST_F(CredentialProviderTest, default_provider_prefers_complete_environment_credentials)
{
  ScopedEnvironment environment;
  environment.set("AWS_ACCESS_KEY_ID", "ASIAENVIRONMENT");
  environment.set("AWS_SECRET_ACCESS_KEY", "environment-secret");
  environment.set("AWS_SESSION_TOKEN", "environment-token");
  environment.set("AWS_EC2_METADATA_DISABLED", "true");

  auto provider = kvikio::make_default_aws_credential_provider("http://127.0.0.1:1");
  auto material = provider->get_auth_material();
  EXPECT_EQ(material->userpwd, "ASIAENVIRONMENT:environment-secret");
  EXPECT_EQ(material->session_token_header,
            std::optional<std::string>{"x-amz-security-token: environment-token"});

  environment.unset("AWS_SECRET_ACCESS_KEY");
  auto partial = kvikio::make_default_aws_credential_provider("http://127.0.0.1:1");
  EXPECT_THROW(partial->get_auth_material(), std::invalid_argument);

  environment.unset("AWS_ACCESS_KEY_ID");
  auto token_only = kvikio::make_default_aws_credential_provider("http://127.0.0.1:1");
  EXPECT_THROW(token_only->get_auth_material(), std::invalid_argument);
}

TEST_F(CredentialProviderTest, instance_profile_uses_imdsv2_and_coalesces_concurrent_callers)
{
  auto const expiration = std::chrono::system_clock::now() + 1h;
  ScriptedHttpServer server{
    imds_success("ASIAINSTANCEA", "instance-secret-a", "instance-token-a", expiration)};
  ScopedEnvironment environment;
  for (auto const* name : {"AWS_ACCESS_KEY_ID",
                           "AWS_SECRET_ACCESS_KEY",
                           "AWS_SESSION_TOKEN",
                           "AWS_EC2_METADATA_DISABLED",
                           "NO_PROXY",
                           "no_proxy"}) {
    environment.unset(name);
  }
  environment.set("HTTP_PROXY", "http://127.0.0.1:1");
  environment.set("HTTPS_PROXY", "http://127.0.0.1:1");
  environment.set("ALL_PROXY", "http://127.0.0.1:1");

  auto provider = kvikio::make_iam_role_aws_credential_provider(server.origin());
  std::vector<std::future<std::shared_ptr<kvikio::AwsAuthMaterial const>>> callers;
  for (int i = 0; i < 12; ++i) {
    callers.push_back(
      std::async(std::launch::async, [provider] { return provider->get_auth_material(); }));
  }
  auto first = callers.front().get();
  for (std::size_t i = 1; i < callers.size(); ++i) {
    EXPECT_EQ(callers[i].get(), first);
  }
  EXPECT_EQ(first->userpwd, "ASIAINSTANCEA:instance-secret-a");
  EXPECT_EQ(server.finish(), "");
  EXPECT_EQ(server.request_count(), 3);
}

TEST_F(CredentialProviderTest, instance_profile_refreshes_before_expiration)
{
  auto const first_expiration = std::chrono::system_clock::now() + 3s;
  std::vector<ExpectedRequest> requests;
  append_requests(requests,
                  imds_success("ASIAINSTANCEA",
                               "instance-secret-a",
                               "instance-token-a",
                               first_expiration,
                               "metadata-token-a"));
  append_requests(requests,
                  imds_success("ASIAINSTANCEB",
                               "instance-secret-b",
                               "instance-token-b",
                               std::chrono::system_clock::now() + 1h,
                               "metadata-token-b"));
  ScriptedHttpServer server{std::move(requests)};
  ScopedEnvironment environment;
  environment.unset("AWS_EC2_METADATA_DISABLED");

  auto provider = kvikio::make_iam_role_aws_credential_provider(server.origin());
  EXPECT_EQ(provider->get_auth_material()->userpwd, "ASIAINSTANCEA:instance-secret-a");
  std::this_thread::sleep_for(1700ms);
  EXPECT_EQ(provider->get_auth_material()->userpwd, "ASIAINSTANCEB:instance-secret-b");
  EXPECT_EQ(server.finish(), "");
}

TEST_F(CredentialProviderTest, refresh_failure_uses_cache_only_until_actual_expiration)
{
  auto const expiration = std::chrono::system_clock::now() + 4s;
  std::vector<ExpectedRequest> requests;
  append_requests(
    requests, imds_success("ASIAINSTANCEA", "instance-secret-a", "instance-token-a", expiration));
  requests.push_back({"PUT", "/latest/api/token", {}, 500, {}});
  ScriptedHttpServer server{std::move(requests)};
  ScopedEnvironment environment;
  environment.unset("AWS_EC2_METADATA_DISABLED");

  auto provider = kvikio::make_iam_role_aws_credential_provider(server.origin());
  auto first    = provider->get_auth_material();
  std::this_thread::sleep_for(2300ms);
  EXPECT_EQ(provider->get_auth_material(), first);

  environment.set("AWS_EC2_METADATA_DISABLED", "true");
  auto const rounded_expiration =
    std::chrono::system_clock::from_time_t(std::chrono::system_clock::to_time_t(expiration));
  std::this_thread::sleep_until(rounded_expiration + 200ms);
  EXPECT_THROW(provider->get_auth_material(), std::runtime_error);
  EXPECT_EQ(server.finish(), "");
}

TEST_F(CredentialProviderTest, instance_profile_rejects_invalid_role_and_credential_documents)
{
  ScopedEnvironment environment;
  environment.unset("AWS_EC2_METADATA_DISABLED");
  {
    ScriptedHttpServer server{{{"PUT",
                                "/latest/api/token",
                                {"X-aws-ec2-metadata-token-ttl-seconds: 21600\r\n"},
                                200,
                                "metadata-token"},
                               {"GET",
                                "/latest/meta-data/iam/security-credentials/",
                                {"X-aws-ec2-metadata-token: metadata-token\r\n"},
                                200,
                                "bad/role\n"}}};
    auto provider = kvikio::make_iam_role_aws_credential_provider(server.origin());
    EXPECT_THROW(provider->get_auth_material(), std::runtime_error);
    EXPECT_EQ(server.finish(), "");
  }
  {
    auto requests        = imds_success("ASIAINSTANCEA",
                                 "do-not-leak-this-secret",
                                 "do-not-leak-this-token",
                                 std::chrono::system_clock::now() + 1h);
    requests.back().body = credential_document("ASIAINSTANCEA",
                                               "do-not-leak-this-secret",
                                               "do-not-leak-this-token",
                                               std::chrono::system_clock::now() + 1h,
                                               "Failure");
    ScriptedHttpServer server{std::move(requests)};
    auto provider = kvikio::make_iam_role_aws_credential_provider(server.origin());
    try {
      std::ignore = provider->get_auth_material();
      FAIL() << "Expected unsuccessful IMDS credential document to fail";
    } catch (std::exception const& error) {
      EXPECT_EQ(std::string_view{error.what()}.find("do-not-leak"), std::string_view::npos);
    }
    EXPECT_EQ(server.finish(), "");
  }
}

class SequencedCredentialProvider final : public kvikio::AwsCredentialProvider {
 public:
  SequencedCredentialProvider()
    : _materials{kvikio::AwsAuthMaterial::create("ASIAREQUESTA", "secret-a", "token-a"),
                 kvikio::AwsAuthMaterial::create("ASIAREQUESTB", "secret-b", "token-b"),
                 kvikio::AwsAuthMaterial::create("ASIAREQUESTC", "secret-c", "token-c")}
  {
  }

  std::shared_ptr<kvikio::AwsAuthMaterial const> get_auth_material() override
  {
    auto const index = _calls.fetch_add(1);
    if (index >= _materials.size()) { throw std::runtime_error{"unexpected credential request"}; }
    return _materials[index];
  }

  [[nodiscard]] std::size_t calls() const { return _calls.load(); }

 private:
  std::array<std::shared_ptr<kvikio::AwsAuthMaterial const>, 3> _materials;
  std::atomic<std::size_t> _calls{};
};

class RestoreRemoteDefaults {
 public:
  RestoreRemoteDefaults()
    : _backend{kvikio::defaults::remote_io_backend()},
      _direct_receive{kvikio::defaults::remote_direct_receive_mode()}
  {
  }

  ~RestoreRemoteDefaults()
  {
    kvikio::defaults::set_remote_io_backend(_backend);
    kvikio::defaults::set_remote_direct_receive_mode(_direct_receive);
  }

 private:
  kvikio::RemoteIOBackend _backend;
  kvikio::RemoteDirectReceiveMode _direct_receive;
};

TEST_F(CredentialProviderTest, s3_endpoint_resolves_credentials_for_head_blocking_and_multi)
{
  ScriptedHttpServer server{
    {{"HEAD",
      "/object",
      {"Credential=ASIAREQUESTA/", "x-amz-security-token: token-a\r\n"},
      200,
      {},
      {},
      4},
     {"GET",
      "/object",
      {"Range: bytes=0-3\r\n", "Credential=ASIAREQUESTB/", "x-amz-security-token: token-b\r\n"},
      206,
      "data",
      {"Content-Range: bytes 0-3/4"}},
     {"GET",
      "/object",
      {"Range: bytes=0-3\r\n", "Credential=ASIAREQUESTC/", "x-amz-security-token: token-c\r\n"},
      206,
      "data",
      {"Content-Range: bytes 0-3/4"}}}};
  ScopedEnvironment environment;
  environment.set("NO_PROXY", "127.0.0.1");
  environment.set("no_proxy", "127.0.0.1");

  auto provider = std::make_shared<SequencedCredentialProvider>();
  auto endpoint = kvikio::S3Endpoint::create_with_credential_provider(
    server.origin() + "/object", "us-east-1", provider);
  kvikio::RemoteHandle handle{std::move(endpoint)};

  RestoreRemoteDefaults const restore_defaults;
  kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::OFF);
  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::EASY_THREADPOOL);
  std::array<char, 4> blocking{};
  EXPECT_EQ(handle.read(blocking.data(), blocking.size(), 0), blocking.size());
  EXPECT_EQ(std::string(blocking.data(), blocking.size()), "data");

  kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
  std::array<char, 4> multi{};
  EXPECT_EQ(handle.pread(multi.data(), multi.size(), 0, multi.size()).get(), multi.size());
  EXPECT_EQ(std::string(multi.data(), multi.size()), "data");
  EXPECT_EQ(provider->calls(), 3);
  EXPECT_EQ(server.finish(), "");
}

}  // namespace
