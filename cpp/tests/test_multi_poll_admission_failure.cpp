/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/shim/libcurl.hpp>

TEST(MultiPollAdmissionFailure, preserves_transfer_ownership_and_original_exception)
{
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_NUM_REACTORS", "1", 1), 0);
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", "per_pread", 1), 0);

  auto aggregate      = std::make_shared<kvikio::detail::RemoteMultiAggregateContext>(1);
  auto future         = aggregate->get_future();
  auto transfer       = std::make_unique<kvikio::detail::RemoteMultiTransfer>();
  transfer->aggregate = aggregate;
  transfer->curl      = std::make_unique<kvikio::CurlHandle>(
    kvikio::LibCurl::instance().get_handle(), __FILE__, "test");

  std::vector<std::unique_ptr<kvikio::detail::RemoteMultiTransfer>> transfers;
  transfers.push_back(std::move(transfer));

  // Fail at the allocation boundary immediately before the in-flight ownership node is reserved.
  // The local transfer must report the original bad_alloc, not disappear as a moved-from emplace
  // argument and surface as broken_promise.
  kvikio::detail::inject_multi_poll_admission_failure_after_for_testing(0);
  EXPECT_NO_THROW(kvikio::detail::MultiReactorPool::instance().submit_pread(std::move(transfers)));
  ASSERT_EQ(future.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_THROW(std::ignore = future.get(), std::bad_alloc);
}
