/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/direct_receive_cuda.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/remote_direct_receive.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/utils.hpp"

namespace {

class OneShotRangeServer {
 public:
  explicit OneShotRangeServer(std::string body) : _body{std::move(body)}
  {
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) { throw std::runtime_error(std::strerror(errno)); }

    int reuse = 1;
    if (::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
      fail_construction();
    }

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(_listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(_listen_fd, 1) != 0) {
      fail_construction();
    }

    socklen_t length = sizeof(address);
    if (::getsockname(_listen_fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      fail_construction();
    }
    _port   = ntohs(address.sin_port);
    _thread = std::thread{[this] { serve(); }};
  }

  ~OneShotRangeServer()
  {
    if (_listen_fd >= 0) {
      ::shutdown(_listen_fd, SHUT_RDWR);
      ::close(_listen_fd);
      _listen_fd = -1;
    }
    if (_thread.joinable()) { _thread.join(); }
  }

  OneShotRangeServer(OneShotRangeServer const&)            = delete;
  OneShotRangeServer& operator=(OneShotRangeServer const&) = delete;

  [[nodiscard]] std::string url() const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + "/object";
  }

 private:
  [[noreturn]] void fail_construction()
  {
    auto const message = std::string{std::strerror(errno)};
    ::close(_listen_fd);
    _listen_fd = -1;
    throw std::runtime_error(message);
  }

  static bool send_all(int fd, std::string_view bytes)
  {
    while (!bytes.empty()) {
      auto const sent = ::send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL);
      if (sent <= 0) { return false; }
      bytes.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
  }

  void serve() noexcept
  {
    auto const client = ::accept(_listen_fd, nullptr, nullptr);
    if (client < 0) { return; }

    std::string request;
    std::array<char, 4096> buffer{};
    while (request.find("\r\n\r\n") == std::string::npos) {
      auto const received = ::recv(client, buffer.data(), buffer.size(), 0);
      if (received <= 0) {
        ::close(client);
        return;
      }
      request.append(buffer.data(), static_cast<std::size_t>(received));
    }

    auto const last   = _body.size() - 1;
    auto const header = std::string{"HTTP/1.1 206 Partial Content\r\nContent-Length: "} +
                        std::to_string(_body.size()) + "\r\nContent-Range: bytes 0-" +
                        std::to_string(last) + "/" + std::to_string(_body.size()) +
                        "\r\nConnection: close\r\n\r\n";
    static_cast<void>(send_all(client, header) && send_all(client, _body));
    ::shutdown(client, SHUT_RDWR);
    ::close(client);
  }

  std::string _body;
  int _listen_fd{-1};
  std::uint16_t _port{};
  std::thread _thread;
};

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
class ScopedBatchSubmissionDeferral {
 public:
  explicit ScopedBatchSubmissionDeferral(std::size_t logical_preads)
  {
    kvikio::detail::defer_multi_poll_direct_receive_submission_until_for_testing(logical_preads);
  }

  ~ScopedBatchSubmissionDeferral()
  {
    // Zero is always a valid reset value, including during stack unwinding.
    kvikio::detail::defer_multi_poll_direct_receive_submission_until_for_testing(0);
  }

  ScopedBatchSubmissionDeferral(ScopedBatchSubmissionDeferral const&)            = delete;
  ScopedBatchSubmissionDeferral& operator=(ScopedBatchSubmissionDeferral const&) = delete;
};
#endif

[[nodiscard]] kvikio::RemoteHandle make_handle(OneShotRangeServer const& server, std::size_t size)
{
  return kvikio::RemoteHandle{std::make_unique<kvikio::HttpEndpoint>(server.url()), size};
}

[[nodiscard]] std::string copy_to_host(kvikio::test::DevBuffer<char> const& source,
                                       std::size_t size)
{
  std::string result(size, '\0');
  KVIKIO_CHECK_CUDA(cudaMemcpy(result.data(), source.ptr, size, cudaMemcpyDeviceToHost));
  return result;
}

struct CompletionResult {
  std::size_t bytes{};
  std::exception_ptr failure;
};

[[nodiscard]] CompletionResult consume_completion(std::future<std::size_t>& completion) noexcept
{
  try {
    return {.bytes = completion.get()};
  } catch (...) {
    return {.failure = std::current_exception()};
  }
}

void expect_reactor_recovery()
{
  std::string const body(193, 'r');
  OneShotRangeServer server{body};
  auto handle = make_handle(server, body.size());
  kvikio::test::DevBuffer<char> output(body.size());
  auto completion   = handle.pread(output.ptr, body.size(), 0, body.size());
  auto const result = consume_completion(completion);
  ASSERT_FALSE(result.failure);
  EXPECT_EQ(result.bytes, body.size());
  EXPECT_EQ(copy_to_host(output, body.size()), body);
}

class MultiPollDirectReceiveBatchTest : public testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    static std::once_flag configured;
    std::call_once(configured, [] {
      if (::setenv("KVIKIO_REMOTE_IO_NUM_REACTORS", "1", 1) != 0 ||
          ::setenv("KVIKIO_REMOTE_IO_REACTOR_DISPATCH", "PER_PREAD", 1) != 0 ||
          ::setenv("KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS", "2", 1) != 0) {
        throw std::runtime_error(std::strerror(errno));
      }
      KVIKIO_CHECK_CUDA(cudaSetDevice(0));
      auto const slot_size = kvikio::detail::DirectReceiveSlotPool::minimum_slot_size();
      kvikio::defaults::set_task_size(slot_size);
      kvikio::defaults::set_remote_io_backend(kvikio::RemoteIOBackend::MULTI_POLL);
      kvikio::defaults::set_remote_direct_receive_mode(kvikio::RemoteDirectReceiveMode::PREFER);
      kvikio::defaults::set_remote_direct_receive_slot_size(slot_size);
      kvikio::defaults::set_remote_direct_receive_max_pinned_bytes(4 * slot_size);
    });
  }

  void SetUp() override { kvikio::reset_remote_direct_receive_stats(); }
};

TEST_F(MultiPollDirectReceiveBatchTest, combines_independent_preads_in_one_kvikio_h2d_batch)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  std::string const first_body(257, 'a');
  std::string const second_body(389, 'b');
  OneShotRangeServer first_server{first_body};
  OneShotRangeServer second_server{second_body};
  auto first_handle  = make_handle(first_server, first_body.size());
  auto second_handle = make_handle(second_server, second_body.size());
  kvikio::test::DevBuffer<char> first_output(first_body.size());
  kvikio::test::DevBuffer<char> second_output(second_body.size());

  ScopedBatchSubmissionDeferral defer{2};
  auto first  = first_handle.pread(first_output.ptr, first_body.size(), 0, first_body.size());
  auto second = second_handle.pread(second_output.ptr, second_body.size(), 0, second_body.size());
  auto const first_result  = consume_completion(first);
  auto const second_result = consume_completion(second);
  ASSERT_FALSE(first_result.failure);
  ASSERT_FALSE(second_result.failure);
  EXPECT_EQ(first_result.bytes, first_body.size());
  EXPECT_EQ(second_result.bytes, second_body.size());
  EXPECT_EQ(copy_to_host(first_output, first_body.size()), first_body);
  EXPECT_EQ(copy_to_host(second_output, second_body.size()), second_body);

  auto const stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(stats.copied_stream_transfers_completed, 2);
  EXPECT_EQ(stats.copied_stream_h2d_bytes, first_body.size() + second_body.size());
  EXPECT_EQ(stats.copied_stream_h2d_batches, 1);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(MultiPollDirectReceiveBatchTest, rejects_an_unsatisfiable_test_deferral)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  EXPECT_THROW(kvikio::detail::defer_multi_poll_direct_receive_submission_until_for_testing(
                 kvikio::detail::direct_receive_max_slots_per_cuda_batch + 1),
               std::invalid_argument);
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(MultiPollDirectReceiveBatchTest, shared_pre_submit_failure_fails_each_owner_and_recovers)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  std::string const body(257, 'f');
  OneShotRangeServer first_server{body};
  OneShotRangeServer second_server{body};
  auto first_handle  = make_handle(first_server, body.size());
  auto second_handle = make_handle(second_server, body.size());
  kvikio::test::DevBuffer<char> first_output(body.size());
  kvikio::test::DevBuffer<char> second_output(body.size());

  ScopedBatchSubmissionDeferral defer{2};
  kvikio::detail::DirectReceiveCudaBatch::inject_pre_submit_allocation_failure_for_testing();
  auto first               = first_handle.pread(first_output.ptr, body.size(), 0, body.size());
  auto second              = second_handle.pread(second_output.ptr, body.size(), 0, body.size());
  auto const first_result  = consume_completion(first);
  auto const second_result = consume_completion(second);
  ASSERT_TRUE(first_result.failure);
  ASSERT_TRUE(second_result.failure);
  EXPECT_THROW(std::rethrow_exception(first_result.failure), std::bad_alloc);
  EXPECT_THROW(std::rethrow_exception(second_result.failure), std::bad_alloc);

  auto const failed_stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(failed_stats.transfers_failed, 2);
  EXPECT_EQ(failed_stats.copied_stream_transfers_completed, 0);
  EXPECT_EQ(failed_stats.copied_stream_h2d_batches, 0);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);

  expect_reactor_recovery();
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

TEST_F(MultiPollDirectReceiveBatchTest, shared_post_enqueue_failure_fails_each_owner_and_recovers)
{
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  std::string const body(257, 'f');
  OneShotRangeServer first_server{body};
  OneShotRangeServer second_server{body};
  auto first_handle  = make_handle(first_server, body.size());
  auto second_handle = make_handle(second_server, body.size());
  kvikio::test::DevBuffer<char> first_output(body.size());
  kvikio::test::DevBuffer<char> second_output(body.size());

  ScopedBatchSubmissionDeferral defer{2};
  kvikio::detail::DirectReceiveCudaBatch::inject_post_enqueue_failure_for_testing();
  auto first               = first_handle.pread(first_output.ptr, body.size(), 0, body.size());
  auto second              = second_handle.pread(second_output.ptr, body.size(), 0, body.size());
  auto const first_result  = consume_completion(first);
  auto const second_result = consume_completion(second);
  ASSERT_TRUE(first_result.failure);
  ASSERT_TRUE(second_result.failure);
  EXPECT_THROW(std::rethrow_exception(first_result.failure), std::runtime_error);
  EXPECT_THROW(std::rethrow_exception(second_result.failure), std::runtime_error);

  auto const failed_stats = kvikio::remote_direct_receive_stats();
  EXPECT_EQ(failed_stats.transfers_failed, 2);
  EXPECT_EQ(failed_stats.copied_stream_transfers_completed, 0);
  EXPECT_EQ(failed_stats.copied_stream_h2d_bytes, 2 * body.size());
  EXPECT_EQ(failed_stats.copied_stream_h2d_batches, 1);
  EXPECT_EQ(kvikio::detail::DirectReceiveSlotPool::instance().snapshot().checked_out_slots, 0);

  expect_reactor_recovery();
#else
  GTEST_SKIP() << "libcurl does not provide caller-owned strict receive support";
#endif
}

}  // namespace
