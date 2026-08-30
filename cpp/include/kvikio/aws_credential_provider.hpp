/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef KVIKIO_LIBCURL_FOUND
#error \
  "cannot include the remote IO API, please build KvikIO with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
#endif

#include <memory>
#include <optional>
#include <string>

namespace kvikio {

/**
 * @brief Immutable AWS SigV4 authentication material for one libcurl request.
 *
 * The session-token header is represented as a string instead of a `curl_slist`. This lets each
 * `CurlHandle` own and compose its complete request-header list.
 */
struct AwsAuthMaterial {
  std::string userpwd;
  std::optional<std::string> session_token_header;

  /**
   * @brief Validate credentials and build libcurl authentication material.
   */
  static std::shared_ptr<AwsAuthMaterial const> create(
    std::string access_key_id,
    std::string secret_access_key,
    std::optional<std::string> session_token = std::nullopt);
};

/**
 * @brief Source of AWS credentials for S3 requests.
 *
 * Implementations may cache and refresh credentials. This function is called while configuring
 * every new easy handle, before the handle is submitted to either the blocking or multi backend.
 */
class AwsCredentialProvider {
 public:
  virtual ~AwsCredentialProvider() = default;

  [[nodiscard]] virtual std::shared_ptr<AwsAuthMaterial const> get_auth_material() = 0;
};

/**
 * @brief Build a provider for explicit, non-refreshing credentials.
 */
std::shared_ptr<AwsCredentialProvider> make_static_aws_credential_provider(
  std::string aws_access_key,
  std::string aws_secret_access_key,
  std::optional<std::string> aws_session_token = std::nullopt);

/**
 * @brief Build a provider matching the legacy S3 endpoint arguments and environment fallback.
 *
 * Explicit values take precedence over `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
 * `AWS_SESSION_TOKEN`. Values are resolved when this function is called, preserving the original
 * endpoint-construction behavior.
 */
std::shared_ptr<AwsCredentialProvider> make_legacy_env_and_args_credential_provider(
  std::optional<std::string> aws_access_key,
  std::optional<std::string> aws_secret_access_key,
  std::optional<std::string> aws_session_token);

/**
 * @brief Build a refreshable EC2 instance-role provider using IMDSv2 only.
 *
 * @param imds_endpoint_override Optional HTTP endpoint used by tests. Otherwise
 * `AWS_EC2_METADATA_SERVICE_ENDPOINT`, then `http://169.254.169.254`, is used.
 */
std::shared_ptr<AwsCredentialProvider> make_iam_role_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override = std::nullopt);

/**
 * @brief Return KvikIO's default AWS credential selection.
 *
 * Complete environment credentials take precedence. With no environment credentials, KvikIO uses
 * refreshable IMDSv2 instance-role credentials. A partial environment configuration is an error.
 * Calls without an endpoint override share one provider and credential cache process-wide.
 * Supplying an endpoint override creates an independent provider, primarily for tests.
 */
std::shared_ptr<AwsCredentialProvider> make_default_aws_credential_provider(
  std::optional<std::string> imds_endpoint_override = std::nullopt);

}  // namespace kvikio
