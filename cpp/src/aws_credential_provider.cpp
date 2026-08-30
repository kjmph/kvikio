/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <curl/curl.h>

#include <kvikio/aws_credential_provider.hpp>
#include <kvikio/detail/env.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/url.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio {
namespace {

using SystemClock = std::chrono::system_clock;

constexpr auto imds_connect_timeout    = std::chrono::seconds{2};
constexpr auto imds_request_timeout    = std::chrono::seconds{5};
constexpr auto imds_token_ttl          = std::chrono::hours{6};
constexpr auto credential_refresh_skew = std::chrono::minutes{5};
constexpr auto failed_refresh_backoff  = std::chrono::seconds{1};

constexpr std::size_t max_imds_token_size       = 4 * 1024;
constexpr std::size_t max_imds_role_response    = 4 * 1024;
constexpr std::size_t max_imds_credential_size  = 64 * 1024;
constexpr std::size_t max_instance_profile_name = 64;

std::optional<std::string> getenv_nonempty(char const* name)
{
  auto const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') { return std::nullopt; }
  return std::string{value};
}

bool metadata_service_disabled()
{
  auto value = getenv_nonempty("AWS_EC2_METADATA_DISABLED");
  if (!value.has_value()) { return false; }
  std::transform(value->begin(), value->end(), value->begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return *value == "true";
}

bool contains_control_character(std::string_view value)
{
  return std::any_of(
    value.begin(), value.end(), [](unsigned char c) { return c < 0x20U || c == 0x7FU; });
}

void validate_credentials(std::string_view access_key,
                          std::string_view secret_key,
                          std::optional<std::string> const& session_token)
{
  KVIKIO_EXPECT(!access_key.empty(),
                "AWS credentials require a non-empty access key ID",
                std::invalid_argument);
  KVIKIO_EXPECT(!secret_key.empty(),
                "AWS credentials require a non-empty secret access key",
                std::invalid_argument);
  KVIKIO_EXPECT(
    access_key.find(':') == std::string_view::npos && !contains_control_character(access_key),
    "AWS access key ID contains an invalid character",
    std::invalid_argument);
  KVIKIO_EXPECT(!contains_control_character(secret_key),
                "AWS secret access key contains an invalid character",
                std::invalid_argument);
  KVIKIO_EXPECT(!session_token.has_value() ||
                  (!session_token->empty() && !contains_control_character(*session_token)),
                "AWS session token must be non-empty and contain no control character",
                std::invalid_argument);
  if (access_key.starts_with("ASIA")) {
    KVIKIO_EXPECT(session_token.has_value(),
                  "AWS session token is required for temporary access key IDs",
                  std::invalid_argument);
  }
}

std::string trim_ascii_whitespace(std::string value)
{
  auto const is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  auto const begin    = std::find_if_not(value.begin(), value.end(), is_space);
  auto const end      = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
  if (begin >= end) { return {}; }
  return std::string{begin, end};
}

void append_utf8(std::string& output, unsigned int codepoint)
{
  KVIKIO_EXPECT(codepoint <= 0x10FFFFU && !(codepoint >= 0xD800U && codepoint <= 0xDFFFU),
                "IMDS returned malformed JSON",
                std::runtime_error);
  if (codepoint <= 0x7FU) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
}

unsigned int parse_hex_quad(std::string_view json, std::size_t& position)
{
  KVIKIO_EXPECT(position + 4 <= json.size(), "IMDS returned malformed JSON", std::runtime_error);
  unsigned int value{};
  for (int i = 0; i < 4; ++i) {
    auto const c = static_cast<unsigned char>(json[position++]);
    value <<= 4U;
    if (c >= '0' && c <= '9') {
      value |= c - '0';
    } else if (c >= 'a' && c <= 'f') {
      value |= 10U + c - 'a';
    } else if (c >= 'A' && c <= 'F') {
      value |= 10U + c - 'A';
    } else {
      KVIKIO_FAIL("IMDS returned malformed JSON", std::runtime_error);
    }
  }
  return value;
}

void skip_json_whitespace(std::string_view json, std::size_t& position)
{
  while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])) != 0) {
    ++position;
  }
}

std::string parse_json_string(std::string_view json, std::size_t& position)
{
  KVIKIO_EXPECT(position < json.size() && json[position] == '"',
                "IMDS returned malformed JSON",
                std::runtime_error);
  ++position;
  std::string output;
  while (position < json.size()) {
    auto const c = static_cast<unsigned char>(json[position++]);
    if (c == '"') { return output; }
    KVIKIO_EXPECT(c >= 0x20U, "IMDS returned malformed JSON", std::runtime_error);
    if (c != '\\') {
      output.push_back(static_cast<char>(c));
      continue;
    }

    KVIKIO_EXPECT(position < json.size(), "IMDS returned malformed JSON", std::runtime_error);
    switch (json[position++]) {
      case '"': output.push_back('"'); break;
      case '\\': output.push_back('\\'); break;
      case '/': output.push_back('/'); break;
      case 'b': output.push_back('\b'); break;
      case 'f': output.push_back('\f'); break;
      case 'n': output.push_back('\n'); break;
      case 'r': output.push_back('\r'); break;
      case 't': output.push_back('\t'); break;
      case 'u': {
        auto codepoint = parse_hex_quad(json, position);
        if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
          KVIKIO_EXPECT(
            position + 2 <= json.size() && json[position] == '\\' && json[position + 1] == 'u',
            "IMDS returned malformed JSON",
            std::runtime_error);
          position += 2;
          auto const low = parse_hex_quad(json, position);
          KVIKIO_EXPECT(
            low >= 0xDC00U && low <= 0xDFFFU, "IMDS returned malformed JSON", std::runtime_error);
          codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
        }
        append_utf8(output, codepoint);
        break;
      }
      default: KVIKIO_FAIL("IMDS returned malformed JSON", std::runtime_error);
    }
  }
  KVIKIO_FAIL("IMDS returned malformed JSON", std::runtime_error);
}

std::unordered_map<std::string, std::string> parse_flat_json_object(std::string_view json)
{
  std::size_t position{};
  skip_json_whitespace(json, position);
  KVIKIO_EXPECT(position < json.size() && json[position++] == '{',
                "IMDS returned malformed JSON",
                std::runtime_error);

  std::unordered_map<std::string, std::string> values;
  skip_json_whitespace(json, position);
  if (position < json.size() && json[position] == '}') {
    ++position;
  } else {
    for (;;) {
      auto key = parse_json_string(json, position);
      skip_json_whitespace(json, position);
      KVIKIO_EXPECT(position < json.size() && json[position++] == ':',
                    "IMDS returned malformed JSON",
                    std::runtime_error);
      skip_json_whitespace(json, position);
      auto value = parse_json_string(json, position);
      KVIKIO_EXPECT(values.emplace(std::move(key), std::move(value)).second,
                    "IMDS returned duplicate JSON fields",
                    std::runtime_error);
      skip_json_whitespace(json, position);
      KVIKIO_EXPECT(position < json.size(), "IMDS returned malformed JSON", std::runtime_error);
      if (json[position] == '}') {
        ++position;
        break;
      }
      KVIKIO_EXPECT(json[position++] == ',', "IMDS returned malformed JSON", std::runtime_error);
      skip_json_whitespace(json, position);
    }
  }
  skip_json_whitespace(json, position);
  KVIKIO_EXPECT(position == json.size(), "IMDS returned malformed JSON", std::runtime_error);
  return values;
}

std::string const& require_json_field(std::unordered_map<std::string, std::string> const& values,
                                      std::string const& key)
{
  auto const found = values.find(key);
  KVIKIO_EXPECT(found != values.end() && !found->second.empty(),
                "IMDS credential response is missing a required field",
                std::runtime_error);
  return found->second;
}

SystemClock::time_point parse_aws_expiration(std::string const& value)
{
  KVIKIO_EXPECT(
    value.size() == 20, "IMDS credential response has an invalid expiration", std::runtime_error);
  std::tm parsed{};
  std::istringstream input{value};
  input >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%SZ");
  KVIKIO_EXPECT(!input.fail() && input.peek() == std::char_traits<char>::eof(),
                "IMDS credential response has an invalid expiration",
                std::runtime_error);
#if defined(_WIN32)
  auto const timestamp = _mkgmtime(&parsed);
#else
  auto const timestamp = timegm(&parsed);
#endif
  KVIKIO_EXPECT(timestamp != static_cast<std::time_t>(-1),
                "IMDS credential response has an invalid expiration",
                std::runtime_error);

  std::tm normalized{};
#if defined(_WIN32)
  gmtime_s(&normalized, &timestamp);
#else
  KVIKIO_EXPECT(gmtime_r(&timestamp, &normalized) != nullptr,
                "IMDS credential response has an invalid expiration",
                std::runtime_error);
#endif
  std::ostringstream round_trip;
  round_trip << std::put_time(&normalized, "%Y-%m-%dT%H:%M:%SZ");
  KVIKIO_EXPECT(round_trip.str() == value,
                "IMDS credential response has an invalid expiration",
                std::runtime_error);
  return SystemClock::from_time_t(timestamp);
}

struct LimitedResponse {
  std::string value;
  std::size_t limit{};
  bool overflow{};
};

std::size_t append_limited_response(char* data, std::size_t size, std::size_t count, void* userdata)
{
  auto& response = *static_cast<LimitedResponse*>(userdata);
  if (size != 0 && count > response.limit / size) {
    response.overflow = true;
    return CURL_WRITEFUNC_ERROR;
  }
  auto const bytes = size * count;
  if (bytes > response.limit - response.value.size()) {
    response.overflow = true;
    return CURL_WRITEFUNC_ERROR;
  }
  response.value.append(data, bytes);
  return bytes;
}

std::string validate_imds_base_url(std::string url)
{
  KVIKIO_EXPECT(!contains_control_character(url),
                "IMDS endpoint contains an invalid character",
                std::invalid_argument);
  while (url.size() > std::string_view{"http://x"}.size() && url.back() == '/') {
    url.pop_back();
  }
  auto const parsed          = detail::UrlParser::parse(url);
  auto const authority_begin = url.find("://") + 3;
  auto const authority_end   = url.find('/', authority_begin);
  auto const user_info       = url.find('@', authority_begin);
  KVIKIO_EXPECT(parsed.scheme == "http" && parsed.host.has_value() && !parsed.host->empty() &&
                  (!parsed.path.has_value() || parsed.path == "/") && !parsed.query.has_value() &&
                  !parsed.fragment.has_value() &&
                  (user_info == std::string::npos ||
                   (authority_end != std::string::npos && user_info > authority_end)),
                "IMDS endpoint must be an HTTP origin without a path, query, or fragment",
                std::invalid_argument);
  return url;
}

std::string resolve_imds_base_url(std::optional<std::string> override_url)
{
  if (override_url.has_value()) {
    KVIKIO_EXPECT(
      !override_url->empty(), "IMDS endpoint override must not be empty", std::invalid_argument);
    return validate_imds_base_url(std::move(*override_url));
  }
  if (auto endpoint = getenv_nonempty("AWS_EC2_METADATA_SERVICE_ENDPOINT")) {
    return validate_imds_base_url(std::move(*endpoint));
  }
  return "http://169.254.169.254";
}

std::string join_imds_path(std::string const& base, std::string_view path)
{
  std::string result = base;
  result.push_back('/');
  result.append(path);
  return result;
}

std::string perform_imds_request(std::string const& url,
                                 std::optional<std::string> request_header,
                                 bool put,
                                 std::size_t response_limit)
{
  auto curl = create_curl_handle();
  curl.setopt(CURLOPT_URL, url.c_str());
  // Instance credentials must never be disclosed to a proxy selected from the process
  // environment. IMDS is a link-local origin and redirects are not part of IMDSv2.
  curl.setopt(CURLOPT_PROXY, "");
  // The process-wide verbose setting prints request headers. Never let it expose an IMDSv2 token.
  curl.setopt(CURLOPT_VERBOSE, 0L);
  curl.setopt(CURLOPT_FOLLOWLOCATION, 0L);
  curl.setopt(CURLOPT_MAXREDIRS, 0L);
#if LIBCURL_VERSION_NUM >= 0x075500
  curl.setopt(CURLOPT_PROTOCOLS_STR, "http");
#else
  curl.setopt(CURLOPT_PROTOCOLS, CURLPROTO_HTTP);
#endif
  curl.setopt(
    CURLOPT_CONNECTTIMEOUT_MS,
    static_cast<long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(imds_connect_timeout).count()));
  curl.setopt(
    CURLOPT_TIMEOUT_MS,
    static_cast<long>(
      std::chrono::duration_cast<std::chrono::milliseconds>(imds_request_timeout).count()));
  if (request_header.has_value()) { curl.append_http_header(*request_header); }
  if (put) {
    curl.setopt(CURLOPT_CUSTOMREQUEST, "PUT");
    curl.setopt(CURLOPT_POSTFIELDS, "");
    curl.setopt(CURLOPT_POSTFIELDSIZE, 0L);
  }

  LimitedResponse response{.limit = response_limit};
  curl.setopt(CURLOPT_WRITEFUNCTION, append_limited_response);
  curl.setopt(CURLOPT_WRITEDATA, &response);
  try {
    curl.perform([&response] {
      response.value.clear();
      response.overflow = false;
    });
  } catch (...) {
    if (response.overflow) {
      KVIKIO_FAIL("IMDS response exceeded its size limit", std::runtime_error);
    }
    throw;
  }
  long status{};
  curl.getinfo(CURLINFO_RESPONSE_CODE, &status);
  KVIKIO_EXPECT(
    status == 200, "IMDS request returned an unexpected HTTP status", std::runtime_error);
  return std::move(response.value);
}

struct ImdsCredentials {
  std::string access_key;
  std::string secret_key;
  std::string session_token;
  SystemClock::time_point expiration;
};

ImdsCredentials fetch_imds_credentials(std::string const& base_url)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto token = trim_ascii_whitespace(perform_imds_request(
    join_imds_path(base_url, "latest/api/token"),
    "X-aws-ec2-metadata-token-ttl-seconds: " +
      std::to_string(std::chrono::duration_cast<std::chrono::seconds>(imds_token_ttl).count()),
    true,
    max_imds_token_size));
  KVIKIO_EXPECT(!token.empty() && !contains_control_character(token),
                "IMDS returned an invalid metadata token",
                std::runtime_error);

  auto const token_header = "X-aws-ec2-metadata-token: " + token;
  auto role_name          = trim_ascii_whitespace(
    perform_imds_request(join_imds_path(base_url, "latest/meta-data/iam/security-credentials/"),
                         token_header,
                         false,
                         max_imds_role_response));
  auto const valid_role_character = [](unsigned char c) {
    return std::isalnum(c) != 0 || c == '+' || c == '=' || c == ',' || c == '.' || c == '@' ||
           c == '_' || c == '-';
  };
  KVIKIO_EXPECT(!role_name.empty() && role_name.size() <= max_instance_profile_name &&
                  std::all_of(role_name.begin(), role_name.end(), valid_role_character),
                "IMDS returned an invalid instance-profile role name",
                std::runtime_error);

  auto const document = perform_imds_request(
    join_imds_path(base_url, "latest/meta-data/iam/security-credentials/" + role_name),
    token_header,
    false,
    max_imds_credential_size);
  auto const fields = parse_flat_json_object(document);
  KVIKIO_EXPECT(require_json_field(fields, "Code") == "Success",
                "IMDS credential response was not successful",
                std::runtime_error);

  ImdsCredentials credentials{
    .access_key    = require_json_field(fields, "AccessKeyId"),
    .secret_key    = require_json_field(fields, "SecretAccessKey"),
    .session_token = require_json_field(fields, "Token"),
    .expiration    = parse_aws_expiration(require_json_field(fields, "Expiration"))};
  validate_credentials(credentials.access_key, credentials.secret_key, credentials.session_token);
  return credentials;
}

class StaticCredentialProvider final : public AwsCredentialProvider {
 public:
  StaticCredentialProvider(std::string access_key,
                           std::string secret_key,
                           std::optional<std::string> session_token)
    : _material{AwsAuthMaterial::create(
        std::move(access_key), std::move(secret_key), std::move(session_token))}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override { return _material; }

 private:
  std::shared_ptr<AwsAuthMaterial const> _material;
};

class IamRoleCredentialProvider final : public AwsCredentialProvider {
 public:
  explicit IamRoleCredentialProvider(std::optional<std::string> endpoint_override)
    : _base_url{resolve_imds_base_url(std::move(endpoint_override))}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::lock_guard lock{_mutex};
    auto const now = SystemClock::now();
    if (_material && now < _refresh_after) { return _material; }
    try {
      KVIKIO_EXPECT(!metadata_service_disabled(),
                    "EC2 instance metadata is disabled by AWS_EC2_METADATA_DISABLED",
                    std::runtime_error);
      auto credentials      = fetch_imds_credentials(_base_url);
      auto const fetched_at = SystemClock::now();
      KVIKIO_EXPECT(credentials.expiration > fetched_at,
                    "IMDS returned expired instance-profile credentials",
                    std::runtime_error);

      auto const lifetime     = credentials.expiration - fetched_at;
      auto const refresh_lead = std::min(
        std::chrono::duration_cast<SystemClock::duration>(credential_refresh_skew), lifetime / 2);
      auto material  = AwsAuthMaterial::create(std::move(credentials.access_key),
                                              std::move(credentials.secret_key),
                                              std::move(credentials.session_token));
      _expiration    = credentials.expiration;
      _refresh_after = _expiration - refresh_lead;
      _material      = std::move(material);
      return _material;
    } catch (...) {
      auto const failed_at = SystemClock::now();
      if (_material && failed_at < _expiration) {
        _refresh_after = std::min(_expiration, failed_at + failed_refresh_backoff);
        return _material;
      }
      throw;
    }
  }

 private:
  std::mutex _mutex;
  std::string _base_url;
  std::shared_ptr<AwsAuthMaterial const> _material;
  SystemClock::time_point _expiration{};
  SystemClock::time_point _refresh_after{};
};

class DefaultCredentialProvider final : public AwsCredentialProvider {
 public:
  explicit DefaultCredentialProvider(std::optional<std::string> endpoint_override)
    : _endpoint_override{std::move(endpoint_override)}
  {
  }

  std::shared_ptr<AwsAuthMaterial const> get_auth_material() override
  {
    std::shared_ptr<AwsCredentialProvider> provider;
    {
      std::lock_guard lock{_mutex};
      if (!_provider) {
        auto access_key    = getenv_nonempty("AWS_ACCESS_KEY_ID");
        auto secret_key    = getenv_nonempty("AWS_SECRET_ACCESS_KEY");
        auto session_token = getenv_nonempty("AWS_SESSION_TOKEN");
        KVIKIO_EXPECT(access_key.has_value() == secret_key.has_value(),
                      "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together",
                      std::invalid_argument);
        KVIKIO_EXPECT(!session_token.has_value() || access_key.has_value(),
                      "AWS_SESSION_TOKEN requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                      std::invalid_argument);
        if (access_key.has_value()) {
          _provider = make_static_aws_credential_provider(
            std::move(*access_key), std::move(*secret_key), std::move(session_token));
        } else {
          KVIKIO_EXPECT(!metadata_service_disabled(),
                        "No AWS environment credentials are set and EC2 instance metadata is "
                        "disabled",
                        std::runtime_error);
          _provider = make_iam_role_aws_credential_provider(std::move(_endpoint_override));
        }
      }
      provider = _provider;
    }
    return provider->get_auth_material();
  }

 private:
  std::mutex _mutex;
  std::optional<std::string> _endpoint_override;
  std::shared_ptr<AwsCredentialProvider> _provider;
};

}  // namespace

std::shared_ptr<AwsAuthMaterial const> AwsAuthMaterial::create(
  std::string access_key_id,
  std::string secret_access_key,
  std::optional<std::string> session_token)
{
  validate_credentials(access_key_id, secret_access_key, session_token);
  auto material     = std::make_shared<AwsAuthMaterial>();
  material->userpwd = std::move(access_key_id);
  material->userpwd.push_back(':');
  material->userpwd.append(secret_access_key);
  if (session_token.has_value()) {
    material->session_token_header = "x-amz-security-token: " + std::move(*session_token);
  }
  return material;
}

std::shared_ptr<AwsCredentialProvider> make_static_aws_credential_provider(
  std::string aws_access_key,
  std::string aws_secret_access_key,
  std::optional<std::string> aws_session_token)
{
  return std::make_shared<StaticCredentialProvider>(
    std::move(aws_access_key), std::move(aws_secret_access_key), std::move(aws_session_token));
}

std::shared_ptr<AwsCredentialProvider> make_legacy_env_and_args_credential_provider(
  std::optional<std::string> aws_access_key,
  std::optional<std::string> aws_secret_access_key,
  std::optional<std::string> aws_session_token)
{
  auto access_key =
    detail::unwrap_or_env(std::move(aws_access_key),
                          "AWS_ACCESS_KEY_ID",
                          "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID isn't set.");
  auto secret_key = detail::unwrap_or_env(
    std::move(aws_secret_access_key),
    "AWS_SECRET_ACCESS_KEY",
    "S3: must provide `aws_secret_access_key` if AWS_SECRET_ACCESS_KEY isn't set.");
  auto session_token = detail::unwrap_or_env(std::move(aws_session_token), "AWS_SESSION_TOKEN");
  return make_static_aws_credential_provider(
    std::move(*access_key), std::move(*secret_key), std::move(session_token));
}

std::shared_ptr<AwsCredentialProvider> make_iam_role_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override)
{
  return std::make_shared<IamRoleCredentialProvider>(std::move(imds_endpoint_override));
}

std::shared_ptr<AwsCredentialProvider> make_default_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override)
{
  if (imds_endpoint_override.has_value()) {
    return std::make_shared<DefaultCredentialProvider>(std::move(imds_endpoint_override));
  }
  // RemoteHandle::open creates one endpoint per object. Sharing this provider prevents every
  // object and every range from maintaining a separate instance-profile credential cache.
  static auto const process_provider = std::make_shared<DefaultCredentialProvider>(std::nullopt);
  return process_provider;
}

}  // namespace kvikio
