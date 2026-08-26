/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <stdexcept>

#include <gtest/gtest.h>

#include <kvikio/detail/multi_poll_reactor.hpp>

TEST(MultiPollConstructionFailure, unwinds_started_reactors_without_terminating)
{
  ASSERT_EQ(setenv("KVIKIO_REMOTE_IO_NUM_REACTORS", "2", 1), 0);
  kvikio::detail::inject_multi_poll_reactor_construction_failure_after_for_testing(1);

  // The complete reactor set is published before startup. Fail after the first reactor owns a
  // running std::thread; pool construction must stop/join it and propagate the original failure.
  EXPECT_THROW(std::ignore = kvikio::detail::MultiReactorPool::instance(), std::bad_alloc);

  // Function-local static initialization is retriable after an exception. The one-shot hook has
  // reset, so a clean two-reactor pool can still be created.
  EXPECT_NO_THROW(std::ignore = kvikio::detail::MultiReactorPool::instance());
  EXPECT_FALSE(kvikio::detail::MultiReactorPool::instance().is_dead());
}
