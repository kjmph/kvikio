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

TEST(MultiPollSubmissionFailure, partial_handoff_returns_a_failed_future)
{
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_NUM_REACTORS", "1", 1), 0);
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", "per_pread", 1), 0);

  constexpr std::size_t transfer_count = 3;
  auto aggregate = std::make_shared<kvikio::detail::RemoteMultiAggregateContext>(transfer_count);
  auto future    = aggregate->get_future();
  std::vector<std::unique_ptr<kvikio::detail::RemoteMultiTransfer>> transfers;
  transfers.reserve(transfer_count);
  for (std::size_t i = 0; i < transfer_count; ++i) {
    auto transfer       = std::make_unique<kvikio::detail::RemoteMultiTransfer>();
    transfer->aggregate = aggregate;
    transfers.push_back(std::move(transfer));
  }

  // Move one transfer into the inbox, then fail the second allocation. submit_pread() must not
  // propagate after ownership is visible to the reactor; every subrange must instead resolve the
  // already-created aggregate future.
  kvikio::detail::inject_multi_poll_submission_failure_after_for_testing(1);
  EXPECT_NO_THROW(kvikio::detail::MultiReactorPool::instance().submit_pread(std::move(transfers)));
  ASSERT_EQ(future.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_THROW(std::ignore = future.get(), std::bad_alloc);
}
