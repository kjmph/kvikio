/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/remote_direct_receive.hpp>

#include "utils/env.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

constexpr std::string_view direct_receive_test_env = "KVIKIO_TEST_REMOTE_DIRECT_RECEIVE";

class RestoreDirectReceiveMode {
 public:
  RestoreDirectReceiveMode() : _mode{kvikio::defaults::remote_direct_receive_mode()} {}
  ~RestoreDirectReceiveMode() { kvikio::defaults::set_remote_direct_receive_mode(_mode); }

 private:
  kvikio::RemoteDirectReceiveMode _mode;
};

}  // namespace

TEST(DefaultsTest, parse_remote_direct_receive_mode)
{
  using Mode = kvikio::RemoteDirectReceiveMode;
  for (auto const& [value, expected] : std::array{std::pair{"off", Mode::OFF},
                                                  std::pair{"PREFER", Mode::PREFER},
                                                  std::pair{"  Require  ", Mode::REQUIRE}}) {
    kvikio::test::EnvVarContext context{{direct_receive_test_env, value}};
    EXPECT_EQ(kvikio::getenv_or(direct_receive_test_env, Mode::OFF), expected);
  }
}

TEST(DefaultsTest, remote_direct_receive_mode_uses_default_when_unset)
{
  using Mode = kvikio::RemoteDirectReceiveMode;
  unsetenv(std::string{direct_receive_test_env}.c_str());
  EXPECT_EQ(kvikio::getenv_or(direct_receive_test_env, Mode::PREFER), Mode::PREFER);
}

TEST(DefaultsTest, invalid_remote_direct_receive_mode_throws)
{
  using Mode = kvikio::RemoteDirectReceiveMode;
  for (auto const& value : {"on", "true", "best_effort", "strict", ""}) {
    kvikio::test::EnvVarContext context{{direct_receive_test_env, value}};
    EXPECT_THROW(kvikio::getenv_or(direct_receive_test_env, Mode::OFF), std::invalid_argument)
      << "value: " << value;
  }
}

TEST(DefaultsTest, remote_direct_receive_mode_round_trip)
{
  using Mode = kvikio::RemoteDirectReceiveMode;
  RestoreDirectReceiveMode const restore;
  kvikio::defaults::set_remote_direct_receive_mode(Mode::PREFER);
  EXPECT_EQ(kvikio::defaults::remote_direct_receive_mode(), Mode::PREFER);
  kvikio::defaults::set_remote_direct_receive_mode(Mode::REQUIRE);
  EXPECT_EQ(kvikio::defaults::remote_direct_receive_mode(), Mode::REQUIRE);
}

TEST(DefaultsTest, parse_compat_mode_str)
{
  {
    std::vector<std::string> inputs{
      "ON", "on", "On", "TRUE", "true", "True", "YES", "yes", "Yes", "1"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::ON);
    }
  }

  {
    std::vector<std::string> inputs{
      "OFF", "off", "oFf", "FALSE", "false", "False", "NO", "no", "No", "0"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::OFF);
    }
  }

  {
    std::vector<std::string> inputs{"AUTO", "auto", "aUtO"};
    for (auto const& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_compat_mode_str(input), kvikio::CompatMode::AUTO);
    }
  }

  {
    std::vector<std::string> inputs{"", "invalidOption", "11", "*&^Yes"};
    for (auto const& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_compat_mode_str(input), std::invalid_argument);
    }
  }
}

TEST(DefaultsTest, parse_http_status_codes)
{
  {
    std::vector<std::string> inputs{
      "429,500", "429, 500", " 429,500", "429,  500", "429 ,500", "429,500 "};
    std::vector<int> expected = {429, 500};
    for (const auto& input : inputs) {
      EXPECT_EQ(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                expected);
    }
  }

  {
    std::vector<std::string> inputs{"429,", ",429", "a,b", "429,,500", "429,1000"};
    for (const auto& input : inputs) {
      EXPECT_THROW(kvikio::detail::parse_http_status_codes("KVIKIO_HTTP_STATUS_CODES", input),
                   std::invalid_argument);
    }
  }
}

TEST(DefaultsTest, alias_for_getenv_or)
{
  // Passed initializer list is empty
  {
    EXPECT_THAT([=] { kvikio::getenv_or({}, 123); },
                ThrowsMessage<std::invalid_argument>(HasSubstr(
                  "`env_var_names` must contain at least one environment variable name")));
  }

  // Non-string env var has an empty value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", ""}};
    EXPECT_THAT(
      [=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, 123); },
      ThrowsMessage<std::invalid_argument>(HasSubstr("unknown config value KVIKIO_TEST_ALIAS=")));
  }

  // Non-string env var and alias have an empty value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS_1", ""},
                                            {"KVIKIO_TEST_ALIAS_2", ""}};
    EXPECT_THAT(
      [=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123); },
      ThrowsMessage<std::invalid_argument>(HasSubstr("unknown config value KVIKIO_TEST_ALIAS_2=")));
  }

  // String env var has an empty value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", ""}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, std::string{"abc"});
    EXPECT_EQ(env_var_name, "KVIKIO_TEST_ALIAS");
    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_found);
  }

  // String env var and alias have an empty value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS_1", ""},
                                            {"KVIKIO_TEST_ALIAS_2", ""}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, std::string{"abc"});
    EXPECT_EQ(env_var_name, "KVIKIO_TEST_ALIAS_2");
    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_found);
  }

  // Env var has already been set by its alias with the same value
  {
    kvikio::test::EnvVarContext env_var_ctx{
      {"KVIKIO_TEST_ALIAS_1", "10"}, {"KVIKIO_TEST_ALIAS_2", "10"}, {"KVIKIO_TEST_ALIAS_3", "10"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2", "KVIKIO_TEST_ALIAS_3"}, 123);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS_3"});
    EXPECT_EQ(result, 10);
    EXPECT_TRUE(has_found);
  }

  // Env var has already been set by its alias with a different value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS_1", "10"},
                                            {"KVIKIO_TEST_ALIAS_2", "20"}};
    EXPECT_THAT([=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123); },
                ThrowsMessage<std::invalid_argument>(HasSubstr(
                  "Environment variable KVIKIO_TEST_ALIAS_2 (20) has already been set by its alias "
                  "KVIKIO_TEST_ALIAS_1 (10) with a different value")));
  }

  // Env var has invalid value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "abc"}};
    EXPECT_THAT([=] { kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, 123); },
                ThrowsMessage<std::invalid_argument>(
                  HasSubstr("unknown config value KVIKIO_TEST_ALIAS=abc")));
  }

  // 1st alias has a set value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS_1", "654.321"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS_1"});
    EXPECT_EQ(result, 654.321);
    EXPECT_TRUE(has_found);
  }

  // 2nd alias has a set value
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS_2", "654.321"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS_2"});
    EXPECT_EQ(result, 654.321);
    EXPECT_TRUE(has_found);
  }

  // Neither alias has a set value
  {
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS_1", "KVIKIO_TEST_ALIAS_2"}, 123.456);
    EXPECT_TRUE(env_var_name.empty());
    EXPECT_EQ(result, 123.456);
    EXPECT_FALSE(has_found);
  }

  // Special type: bool
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "yes"}};
    auto const [env_var_name, result, has_found] = kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, false);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    EXPECT_TRUE(result);
    EXPECT_TRUE(has_found);
  }
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "OFF"}};
    auto const [env_var_name, result, has_found] = kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, false);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    EXPECT_FALSE(result);
    EXPECT_TRUE(has_found);
  }

  // Special type: CompatMode
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "yes"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, kvikio::CompatMode::AUTO);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    EXPECT_EQ(result, kvikio::CompatMode::ON);
    EXPECT_TRUE(has_found);
  }
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "FALSE"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, kvikio::CompatMode::AUTO);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    EXPECT_EQ(result, kvikio::CompatMode::OFF);
    EXPECT_TRUE(has_found);
  }
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "aUtO"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, kvikio::CompatMode::ON);
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    EXPECT_EQ(result, kvikio::CompatMode::AUTO);
    EXPECT_TRUE(has_found);
  }

  // Special type: std::vector<int>
  {
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_TEST_ALIAS", "109, 108, 107"}};
    auto const [env_var_name, result, has_found] =
      kvikio::getenv_or({"KVIKIO_TEST_ALIAS"}, std::vector<int>{111, 112, 113});
    EXPECT_EQ(env_var_name, std::string_view{"KVIKIO_TEST_ALIAS"});
    std::vector<int> expected{109, 108, 107};
    EXPECT_EQ(result, expected);
    EXPECT_TRUE(has_found);
  }
}
