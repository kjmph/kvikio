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

TEST(MultiPollDispatchFailure, later_reactor_failure_cannot_start_an_earlier_bucket)
{
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_NUM_REACTORS", "2", 1), 0);
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", "per_chunk", 1), 0);

  constexpr std::size_t transfer_count = 2;
  auto aggregate = std::make_shared<kvikio::detail::RemoteMultiAggregateContext>(transfer_count);
  auto future    = aggregate->get_future();
  std::vector<std::unique_ptr<kvikio::detail::RemoteMultiTransfer>> transfers;
  transfers.reserve(transfer_count);
  for (std::size_t i = 0; i < transfer_count; ++i) {
    auto transfer       = std::make_unique<kvikio::detail::RemoteMultiTransfer>();
    transfer->aggregate = aggregate;
    transfers.push_back(std::move(transfer));
  }

  // PER_CHUNK sends the first transfer to reactor 0 and the second to reactor 1. Failing the
  // second handoff must mark pool death while the dispatch gate still prevents reactor 0 from
  // draining its earlier bucket.
  kvikio::detail::inject_multi_poll_submission_failure_after_for_testing(1);
  EXPECT_NO_THROW(kvikio::detail::MultiReactorPool::instance().submit_pread(std::move(transfers)));
  ASSERT_EQ(future.wait_for(std::chrono::seconds{5}), std::future_status::ready);
  EXPECT_THROW(std::ignore = future.get(), std::bad_alloc);
}
