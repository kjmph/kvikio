/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/detail/multi_poll_reactor.hpp>
#include <kvikio/detail/stream.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/logger_macros.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

namespace {

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
constexpr std::size_t failure_injection_disabled = std::numeric_limits<std::size_t>::max();
std::atomic<std::size_t> submission_failure_countdown{failure_injection_disabled};
std::atomic<std::size_t> admission_failure_countdown{failure_injection_disabled};
std::atomic<std::size_t> reactor_construction_failure_countdown{failure_injection_disabled};

void maybe_inject_failure(std::atomic<std::size_t>& countdown)
{
  auto remaining = countdown.load(std::memory_order_relaxed);
  while (remaining != failure_injection_disabled) {
    if (remaining == 0) {
      if (countdown.compare_exchange_weak(remaining,
                                          failure_injection_disabled,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) {
        throw std::bad_alloc{};
      }
    } else if (countdown.compare_exchange_weak(
                 remaining, remaining - 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return;
    }
  }
}
#endif

void log_failure_noexcept(char const* prefix, std::exception_ptr failure) noexcept
{
  try {
    if (failure != nullptr) { std::rethrow_exception(failure); }
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + "unknown exception");
    } catch (...) {
    }
  } catch (std::exception const& error) {
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + error.what());
    } catch (...) {
    }
  } catch (...) {
    try {
      KVIKIO_LOG_ERROR(std::string{prefix} + "unknown exception");
    } catch (...) {
    }
  }
}

void log_message_noexcept(char const* prefix, char const* message) noexcept
{
  try {
    KVIKIO_LOG_ERROR(std::string{prefix} + message);
  } catch (...) {
  }
}

void fail_aggregate_transfer(RemoteMultiTransfer& transfer, std::exception_ptr failure) noexcept
{
  if (transfer.aggregate_terminal_reported) { return; }
  transfer.aggregate_terminal_reported = true;
  try {
    transfer.aggregate->on_subrange_failed(std::move(failure));
  } catch (...) {
    // Cleanup must resolve every transfer owned by this reactor even if a broken promise or an
    // unexpected aggregate callback failure makes one notification throw.
    log_failure_noexcept("remote transfer aggregate failure callback threw: ",
                         std::current_exception());
  }
}

}  // namespace

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
void inject_multi_poll_submission_failure_after_for_testing(
  std::size_t successful_insertions) noexcept
{
  submission_failure_countdown.store(successful_insertions, std::memory_order_relaxed);
}

void inject_multi_poll_admission_failure_after_for_testing(
  std::size_t successful_admissions) noexcept
{
  admission_failure_countdown.store(successful_admissions, std::memory_order_relaxed);
}

void inject_multi_poll_reactor_construction_failure_after_for_testing(
  std::size_t successful_reactors) noexcept
{
  reactor_construction_failure_countdown.store(successful_reactors, std::memory_order_relaxed);
}
#endif

CurlMultiAttachment::CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept
  : _multi{multi}, _easy{easy}
{
}

void CurlMultiAttachment::reset() noexcept
{
  if (_multi != nullptr && _easy != nullptr) {
    // Best-effort detach on the reactor I/O thread. If curl_multi_remove_handle fails (rare), the
    // handle stays attached and the owning CurlHandle still returns it to the LibCurl pool, which
    // is undefined behavior in libcurl. There is no better recovery available here.
    auto const mc = curl_multi_remove_handle(_multi, _easy);
    if (mc != CURLM_OK) {
      log_message_noexcept("CurlMultiAttachment: curl_multi_remove_handle failed: ",
                           curl_multi_strerror(mc));
    }
  }
  _multi = nullptr;
  _easy  = nullptr;
}

CurlMultiAttachment::~CurlMultiAttachment() { reset(); }

CurlMultiAttachment::CurlMultiAttachment(CurlMultiAttachment&& other) noexcept
  : _multi{std::exchange(other._multi, nullptr)}, _easy{std::exchange(other._easy, nullptr)}
{
}

CurlMultiAttachment& CurlMultiAttachment::operator=(CurlMultiAttachment&& other) noexcept
{
  if (this != &other) {
    // Detach whatever this guard currently holds before taking over o's handle.
    reset();
    _multi = std::exchange(other._multi, nullptr);
    _easy  = std::exchange(other._easy, nullptr);
  }
  return *this;
}

RemoteMultiTransfer::~RemoteMultiTransfer()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  // A device transfer still holding its bounce buffer reaches here only on a failure path. The
  // success path moves the buffer into recycle_after, leaving buffer.get() == nullptr.
  if (!is_device || buffer.get() == nullptr) { return; }
  try {
    PushAndPopContext c(device_ctx);
    BounceBufferCache::instance().recycle_now(device_ctx, std::move(buffer));
  } catch (...) {
    log_failure_noexcept("RemoteMultiTransfer: buffer recycle failed: ", std::current_exception());
  }
}

RemoteMultiAggregateContext::RemoteMultiAggregateContext(std::size_t num_subranges)
  : _subranges_left{num_subranges}
{
  KVIKIO_EXPECT(num_subranges > 0,
                "RemoteMultiAggregateContext requires at least one sub-range",
                std::invalid_argument);
}

void RemoteMultiAggregateContext::on_subrange_complete(std::size_t bytes)
{
  _total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  // The last thread to decrement _subranges_left to zero fulfills the promise. Its acq_rel
  // decrement acquires every other thread's relaxed _total_bytes writes (each released by that
  // thread's own decrement), so the sum is complete. _first_exception needs no ordering here, since
  // it is written and read under _exception_mutex.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    // Finish the observation before fulfilling the promise below. The other order would let the
    // caller return from `future.get()` before the observation had been delivered.
    if (recorder) {
      if (_first_exception) {
        recorder->finish_with_failure();
      } else {
        recorder->finish(_total_bytes.load(std::memory_order_relaxed));
      }
    }
    if (_first_exception) {
      _promise.set_exception(_first_exception);
    } else {
      _promise.set_value(_total_bytes.load(std::memory_order_relaxed));
    }
  }
}

void RemoteMultiAggregateContext::on_subrange_failed(std::exception_ptr eptr)
{
  {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (!_first_exception) { _first_exception = eptr; }
  }
  // Last thread to decrement to zero fulfills the promise.
  if (_subranges_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> const lock(_exception_mutex);
    if (recorder) { recorder->finish_with_failure(); }
    _promise.set_exception(_first_exception);
  }
}

std::future<std::size_t> RemoteMultiAggregateContext::get_future() { return _promise.get_future(); }

std::optional<std::size_t> reactor_concurrency_limit(std::size_t max_total,
                                                     std::size_t num_reactors,
                                                     std::size_t reactor_index)
{
  KVIKIO_EXPECT(
    num_reactors > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);
  KVIKIO_EXPECT(reactor_index < num_reactors,
                "reactor index must be smaller than remote_io_num_reactors",
                std::invalid_argument);
  if (max_total == 0) { return std::nullopt; }
  KVIKIO_EXPECT(max_total >= num_reactors,
                "remote_io_max_concurrent_requests must be zero (unlimited) or at least "
                "remote_io_num_reactors",
                std::invalid_argument);

  auto const base      = max_total / num_reactors;
  auto const remainder = max_total % num_reactors;
  return base + static_cast<std::size_t>(reactor_index < remainder);
}

MultiPollReactor::MultiPollReactor(MultiReactorPool* pool,
                                   std::optional<std::size_t> max_concurrent_requests)
  : _pool{pool}, _request_limiter{max_concurrent_requests}
{
  KVIKIO_EXPECT(
    _pool != nullptr, "MultiPollReactor requires a non-null pool", std::invalid_argument);
  // Force LibCurl global init before we create the multi handle.
  std::ignore = LibCurl::instance();
  try {
    _curl_multi = curl_multi_init();
    KVIKIO_EXPECT(_curl_multi != nullptr, "curl_multi_init() failed", std::runtime_error);
    set_connection_cache_size(max_concurrent_requests);
  } catch (...) {
    if (_curl_multi != nullptr) {
      std::ignore = curl_multi_cleanup(_curl_multi);
      _curl_multi = nullptr;
    }
    throw;
  }
}

std::optional<long> connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) noexcept
{
  if (!max_concurrent_requests.has_value()) { return std::nullopt; }

  // libcurl documents this option as taking a `long`, and the value is internally stored as an
  // `unsigned int`. So we cap at whichever of UINT_MAX and LONG_MAX is smaller.
  constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());
  constexpr auto long_max = static_cast<std::size_t>(std::numeric_limits<long>::max());
  constexpr std::size_t max_settable = std::min(uint_max, long_max);

  // min(max_concurrent_requests * headroom_scale, max_settable), with int overflow avoidance
  constexpr std::size_t headroom_scale = 4;
  auto const max_req_adjusted          = std::max<std::size_t>(max_concurrent_requests.value(), 1);
  auto const tmp = std::min<std::size_t>(max_req_adjusted, max_settable / headroom_scale);
  return static_cast<long>(tmp * headroom_scale);
}

void MultiPollReactor::set_connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) const
{
  auto const cache_size = connection_cache_size(max_concurrent_requests);
  if (!cache_size.has_value()) { return; }

  auto const mc = curl_multi_setopt(_curl_multi, CURLMOPT_MAXCONNECTS, cache_size.value());
  KVIKIO_EXPECT(mc == CURLM_OK,
                std::string("curl_multi_setopt(CURLMOPT_MAXCONNECTS): ") + curl_multi_strerror(mc),
                std::runtime_error);
}

MultiPollReactor::~MultiPollReactor() noexcept
{
  // The process-wide pool is intentionally leaked during normal operation, but this destructor is
  // required when pool construction unwinds after a later reactor fails. Stop and join the thread
  // before member destruction so resource exhaustion propagates as an exception instead of
  // std::thread::~thread calling std::terminate.
  stop();
  if (_curl_multi != nullptr) {
    std::ignore = curl_multi_cleanup(_curl_multi);
    _curl_multi = nullptr;
  }
}

void MultiPollReactor::start()
{
  KVIKIO_EXPECT(!_io_thread.joinable(), "MultiPollReactor is already started", std::logic_error);
  _io_thread = std::thread(&MultiPollReactor::io_thread_main, this);
}

void MultiPollReactor::stop() noexcept
{
  _stop.store(true, std::memory_order_release);
  wakeup();
  if (_io_thread.joinable()) { _io_thread.join(); }
}

void MultiPollReactor::wakeup() noexcept { std::ignore = curl_multi_wakeup(_curl_multi); }

void MultiPollReactor::submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers) noexcept
{
  if (transfers.empty()) { return; }
  std::exception_ptr fail_reason;
  try {
    {
      std::lock_guard<std::mutex> const lock(_submit_mutex);
      if (_pool->is_dead()) {
        // The pool is dead. Fail the batch immediately instead of pushing into an inbox that will
        // never be drained.
        fail_reason = _pool->death_reason();
      } else {
        try {
          for (auto& transfer : transfers) {
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
            maybe_inject_failure(submission_failure_countdown);
#endif
            _inbox.push_back(std::move(transfer));
          }
        } catch (...) {
          // Publish pool death while the inbox lock is still held. Otherwise this reactor could
          // drain a partially inserted batch in the window between releasing the lock and marking
          // the pool dead.
          fail_reason = std::current_exception();
          _pool->signal_death(fail_reason);
          // Another reactor may have won the race to declare pool death. Every transfer must
          // observe that first, canonical failure rather than this later local failure.
          fail_reason = _pool->death_reason();
        }
      }
    }
  } catch (...) {
    // A submit-mutex failure happens before this call can safely publish ownership. Treat it as a
    // pool failure as well so this noexcept boundary always resolves the caller's future.
    fail_reason = std::current_exception();
    _pool->signal_death(fail_reason);
    fail_reason = _pool->death_reason();
  }
  if (fail_reason) {
    for (auto& transfer : transfers) {
      if (transfer) { fail_aggregate_transfer(*transfer, fail_reason); }
    }
    return;
  }
  wakeup();
}

void MultiPollReactor::io_thread_main()
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;
  try {
    while (!_stop.load(std::memory_order_acquire) && !_pool->is_dead()) {
      // Stage (1): Splice newly submitted transfers out of the inbox (shared by the reactor thread
      // and submission thread) to minimize the lock duration.
      bool submission_failed_pool{};
      {
        // Match MultiReactorPool::submit_pread's lock order. This prevents any reactor from
        // draining an early PER_CHUNK bucket until every bucket from that pread has been handed off
        // successfully (or the pool has been marked dead).
        std::lock_guard<std::mutex> const submission_lock(_pool->_submission_mutex);
        std::lock_guard<std::mutex> const inbox_lock(_submit_mutex);
        submission_failed_pool = _pool->is_dead();
        if (!submission_failed_pool) {
          if (_pending.empty()) {
            std::swap(_pending, _inbox);
          } else {
            while (!_inbox.empty()) {
              _pending.push_back(std::move(_inbox.front()));
              _inbox.pop_front();
            }
          }
        }
      }
      if (submission_failed_pool) { break; }

      // Iterate the per-reactor _pending: Each entry is either admitted to libcurl or moved to
      // `deferred_transfers`, which becomes the new `_pending` at the end.
      std::deque<std::unique_ptr<RemoteMultiTransfer>> deferred_transfers;
      // Contexts whose bounce-buffer shard has already missed during this walk. It is assumed that
      // distinct contexts are few, so a flat vector with linear find suffices.
      std::vector<CUcontext> exhausted_ctxs;
      // Earliest backoff deadline among the retried transfers.
      std::optional<std::chrono::steady_clock::time_point> earliest_ready_at;
      // Whether anything is deferred because a limiter slot or bounce buffer is unavailable, rather
      // than because its backoff has not elapsed for retry.
      bool deferred_for_resource = false;
      auto const walk_start      = std::chrono::steady_clock::now();
      while (!_pending.empty()) {
        auto transfer = std::move(_pending.front());
        _pending.pop_front();
        decltype(_in_flight)::iterator in_flight_slot = _in_flight.end();
        bool in_flight_slot_reserved{};
        try {
          // Defer a transfer if it is still serving its backoff for retry.
          if (transfer->ready_at > walk_start) {
            if (earliest_ready_at.has_value()) {
              earliest_ready_at = std::min(earliest_ready_at.value(), transfer->ready_at);
            } else {
              earliest_ready_at = transfer->ready_at;
            }
            deferred_transfers.push_back(std::move(transfer));
            continue;
          }

          // This ctx already missed the cache this walk, so defer without taking a limiter slot. At
          // worst this is pessimistic by one iteration if a recycle frees a buffer mid-walk.
          if (transfer->is_device &&
              std::find(exhausted_ctxs.begin(), exhausted_ctxs.end(), transfer->device_ctx) !=
                exhausted_ctxs.end()) {
            deferred_for_resource = true;
            deferred_transfers.push_back(std::move(transfer));
            continue;
          }

          // Gate 1 caps network concurrency. Limit the HTTP range requests attached to this
          // reactor's multi handle at once, host and device combined.
          auto slot = _request_limiter.try_acquire();
          if (!slot) {
            deferred_for_resource = true;
            deferred_transfers.push_back(std::move(transfer));
            while (!_pending.empty()) {
              deferred_transfers.push_back(std::move(_pending.front()));
              _pending.pop_front();
            }
            break;
          }

          if (transfer->is_device) {
            // Gate 2 caps bounce-buffer use per (reactor thread, CUDA context) across all pipeline
            // phases. A limiter slot freed at libcurl completion does not free the buffer, which
            // stays in-flight until the H2D drains and the recycle callback fires.
            std::optional<CudaPinnedBounceBufferPool::Buffer> bounce_buffer;
            {
              PushAndPopContext c(transfer->device_ctx);
              bounce_buffer = BounceBufferCache::instance().try_get(transfer->device_ctx);
            }
            if (!bounce_buffer.has_value()) {
              deferred_for_resource = true;
              exhausted_ctxs.push_back(transfer->device_ctx);
              deferred_transfers.push_back(std::move(transfer));
              continue;
            }
            transfer->buffer            = std::move(bounce_buffer.value());
            transfer->ctx.pinned_buffer = transfer->buffer.get();
          }

          CURL* easy = transfer->curl->handle();
          // Allocate the ownership node before attaching the easy handle. If allocation or rehash
          // fails, `transfer` is still locally owned and can be failed with the original exception
          // instead of being lost as a moved-from argument to unordered_map::emplace.
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
          maybe_inject_failure(admission_failure_countdown);
#endif
          auto const insertion = _in_flight.try_emplace(easy, nullptr);
          KVIKIO_EXPECT(insertion.second,
                        "MultiPollReactor: duplicate easy handle admission",
                        std::logic_error);
          in_flight_slot          = insertion.first;
          in_flight_slot_reserved = true;

          auto const mc = curl_multi_add_handle(_curl_multi, easy);
          if (mc != CURLM_OK) {
            KVIKIO_FAIL(std::string("curl_multi_add_handle: ") + curl_multi_strerror(mc),
                        std::runtime_error);
          }
          transfer->attachment    = CurlMultiAttachment{_curl_multi, easy};
          transfer->slot          = std::move(slot);
          in_flight_slot->second  = std::move(transfer);
          in_flight_slot_reserved = false;
        } catch (...) {
          // Establish (or observe) the pool's canonical first failure before resolving any
          // aggregate. Otherwise a concurrent reactor could win pool death after these locally
          // owned transfers had permanently recorded this later exception.
          _pool->signal_death(std::current_exception());
          auto const failure = _pool->death_reason();
          if (in_flight_slot_reserved) { _in_flight.erase(in_flight_slot); }

          // Do not allocate while recovering from an allocation failure. Fail locally owned state
          // in place; the outer reactor catch declares pool death and fail_all_pending handles the
          // entries that remain in reactor containers.
          if (transfer) { fail_aggregate_transfer(*transfer, failure); }
          for (auto& deferred : deferred_transfers) {
            if (deferred) { fail_aggregate_transfer(*deferred, failure); }
          }
          std::rethrow_exception(failure);
        }
      }
      // The walk drained `_pending`. The deferred entries become the new pending queue.
      std::swap(_pending, deferred_transfers);

      // Stage (2): Drive transfers in a non-blocking way.
      int running_handles   = 0;
      auto const perform_mc = curl_multi_perform(_curl_multi, &running_handles);
      KVIKIO_EXPECT(perform_mc == CURLM_OK,
                    std::string("curl_multi_perform: ") + curl_multi_strerror(perform_mc),
                    std::runtime_error);

      // Stage (3): Drain completions.
      int msgs_left = 0;
      // A completion frees a limiter slot, which may unblock a deferred transfer waiting on one.
      // Stage (4) uses this to shorten the poll timeout.
      bool completed_any = false;
      while (auto* msg = curl_multi_info_read(_curl_multi, &msgs_left)) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        completed_any = true;
        auto* easy    = msg->easy_handle;
        auto res      = msg->data.result;

        auto it = _in_flight.find(easy);
        KVIKIO_EXPECT(it != _in_flight.end(),
                      "MultiPollReactor: completion for unknown handle",
                      std::runtime_error);
        auto transfer = std::move(it->second);
        _in_flight.erase(it);

        std::exception_ptr transfer_err;
        try {
          if (res == CURLE_OK && !transfer->ctx.overflow_error) {
            if (transfer->is_device) {
              // Phase A (network -> pinned) done. Now schedule Phase B (pinned -> device) on this
              // (thread, ctx) stream and hand the buffer to a cuLaunchHostFunc recycle callback so
              // the cache slot is returned when the H2D drains.
              PushAndPopContext c(transfer->device_ctx);
              CUstream stream = StreamCachePerThreadAndContext::get();
              bool h2d_may_be_enqueued{false};
              try {
                h2d_may_be_enqueued = true;
                KVIKIO_CUDA_DRIVER_TRY(
                  cudaAPI::instance().MemcpyHtoDAsync(convert_void2deviceptr(transfer->device_dst),
                                                      transfer->buffer.get(),
                                                      transfer->ctx.size,
                                                      stream));
                transfer->aggregate->io_event_barrier->record_event(stream);
                BounceBufferCache::instance().recycle_after(transfer->device_ctx,
                                                            std::move(transfer->buffer),
                                                            stream,
                                                            [curl_multi = _curl_multi]() noexcept {
                                                              std::ignore =
                                                                curl_multi_wakeup(curl_multi);
                                                            });
              } catch (...) {
                // If event/callback setup failed after an H2D may have been queued and this object
                // still owns the source slot, drain the stream before normal failure cleanup can
                // recycle it. recycle_after handles the moved-buffer case internally.
                if (h2d_may_be_enqueued && transfer->buffer.get() != nullptr) {
                  try {
                    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
                  } catch (...) {
                    auto const sync_error = std::current_exception();
                    transfer->aggregate->io_event_barrier->mark_completion_unknown();
                    BounceBufferCache::instance().abandon_checked_out_after_failed_sync(
                      transfer->device_ctx, std::move(transfer->buffer));
                    log_failure_noexcept("H2D failure synchronization failed: ", sync_error);
                  }
                }
                throw;
              }
            }
            transfer->aggregate->on_subrange_complete(transfer->ctx.size);
          } else if (transfer->ctx.overflow_error) {
            // Prefer the handle's recorded error buffer. Fall back to the generic strerror text
            // when libcurl recorded no message.
            auto const errmsg = transfer->curl->error_message();
            std::string desc  = std::string("curl_multi transfer failed (") +
                               (errmsg.empty() ? std::string{curl_easy_strerror(res)} : errmsg) +
                               ") [server returned more bytes than requested; maybe range support "
                               "missing?]";
            transfer_err = std::make_exception_ptr(std::runtime_error(std::move(desc)));
          } else {
            long http_code = 0;
            transfer->curl->getinfo(CURLINFO_RESPONSE_CODE, &http_code);
            ++transfer->attempt;
            auto const errmsg  = transfer->curl->error_message();
            auto const outcome = transfer->retry_policy->evaluate(
              res, http_code, transfer->attempt, errmsg, "curl_multi transfer failed");

            if (outcome.decision == RetryDecision::RETRY) {
              try {
                KVIKIO_LOG_WARN(outcome.message);
              } catch (...) {
                // Retry diagnostics must not turn an otherwise recoverable request into failure.
              }
              auto const ready_at = std::chrono::steady_clock::now() + outcome.delay_ms;
              // If a shorter backoff appears
              if (earliest_ready_at.has_value()) {
                earliest_ready_at = std::min(earliest_ready_at.value(), ready_at);
              } else {
                earliest_ready_at = ready_at;
              }
              requeue_for_retry(std::move(transfer), ready_at);
              continue;
            }

            transfer_err = std::make_exception_ptr(std::runtime_error(outcome.message));
          }
        } catch (...) {
          transfer_err = std::current_exception();
        }
        if (transfer_err) { fail_aggregate_transfer(*transfer, transfer_err); }
      }

      // Stage (4): Wait for socket activity, a wakeup, a timeout, or elapsed backoff for retry.
      constexpr int idle_timeout_ms = 1000;
      constexpr int busy_timeout_ms = 10;
      int poll_timeout_ms{};
      if (_pending.empty()) {
        // Nothing queued
        poll_timeout_ms = idle_timeout_ms;
      } else if (!deferred_for_resource && earliest_ready_at.has_value()) {
        // Wait for the earliest elapsed backoff, not a limiter slot or bounce buffer resource
        auto const wait_ms = std::chrono::ceil<std::chrono::milliseconds>(
                               earliest_ready_at.value() - std::chrono::steady_clock::now())
                               .count();
        if (wait_ms <= 0) {
          poll_timeout_ms = 0;
        } else if (wait_ms >= idle_timeout_ms) {
          poll_timeout_ms = idle_timeout_ms;
        } else {
          poll_timeout_ms = static_cast<int>(wait_ms);
        }
      } else if (completed_any) {
        // A transfer completion frees the resource a queued transfer needs, so re-admit at once.
        poll_timeout_ms = 0;
      } else {
        // Wait for a limiter slot or bounce buffer resource
        poll_timeout_ms = busy_timeout_ms;
      }
      auto const poll_mc = curl_multi_poll(_curl_multi,
                                           nullptr,          // extra_fds
                                           0,                // extra_nfds
                                           poll_timeout_ms,  // timeout_ms
                                           nullptr);         // numfds
      KVIKIO_EXPECT(poll_mc == CURLM_OK,
                    std::string("curl_multi_poll: ") + curl_multi_strerror(poll_mc),
                    std::runtime_error);
    }
  } catch (...) {
    // Any fatal reactor-loop error caught above declares pool-wide death. The first reactor to
    // signal wins. Subsequent signals are silently ignored.
    auto const failure = std::current_exception();
    log_failure_noexcept("MultiPollReactor fatal error; reactor pool declared dead: ", failure);
    _pool->signal_death(failure);
  }
  // `_stop` is used only while unwinding pool construction. Such a reactor has never been published
  // to submit_pread(), hence owns no transfers and has no failure to fan out. In every operational
  // exit the pool is dead and carries the non-null exception used to resolve callers' futures.
  if (_stop.load(std::memory_order_acquire) && !_pool->is_dead()) { return; }

  // Reached by catching the exception above or by noticing _pool->is_dead() at the loop top. Drain
  // our own state with the recorded reason so no caller's future.get() hangs.
  fail_all_pending(_pool->death_reason());
}

void MultiPollReactor::requeue_for_retry(std::unique_ptr<RemoteMultiTransfer> transfer,
                                         std::chrono::steady_clock::time_point ready_at) noexcept
{
  using BounceBufferCache = BounceBufferCachePerThreadAndContext<CudaPinnedAllocator>;

  // Extend the lifetime of aggregate (a shared pointer).
  auto aggregate = transfer->aggregate;

  try {
    transfer->attachment.reset();
    transfer->slot.reset();

    if (transfer->is_device && transfer->buffer.get() != nullptr) {
      PushAndPopContext c(transfer->device_ctx);
      BounceBufferCache::instance().recycle_now(transfer->device_ctx, std::move(transfer->buffer));
      transfer->ctx.pinned_buffer = nullptr;
    }

    transfer->ctx.reset_for_retry();
    transfer->curl->clear_error_message();
    transfer->ready_at = ready_at;
    _pending.push_back(std::move(transfer));
  } catch (...) {
    auto const failure = std::current_exception();
    if (transfer) {
      fail_aggregate_transfer(*transfer, failure);
    } else {
      // A successful deque insertion is the final operation in the try block, so this branch is
      // defensive only. Never let a broken promise or recorder callback escape noexcept.
      try {
        aggregate->on_subrange_failed(failure);
      } catch (...) {
        log_failure_noexcept("remote retry aggregate failure callback threw: ",
                             std::current_exception());
      }
    }
  }
}

void MultiPollReactor::fail_all_pending(std::exception_ptr eptr)
{
  // Drain the inbox under the submit mutex.
  {
    std::lock_guard<std::mutex> const lock(_submit_mutex);
    while (!_inbox.empty()) {
      auto transfer = std::move(_inbox.front());
      _inbox.pop_front();
      fail_aggregate_transfer(*transfer, eptr);
    }
  }

  // Drain the deferred queue.
  while (!_pending.empty()) {
    auto transfer = std::move(_pending.front());
    _pending.pop_front();
    fail_aggregate_transfer(*transfer, eptr);
  }

  // In-flight is touched only by the I/O thread, which is us, so no lock needed.
  for (auto& in_flight_entry : _in_flight) {
    fail_aggregate_transfer(*in_flight_entry.second, eptr);
  }
  _in_flight.clear();
}

MultiReactorPool::MultiReactorPool() : _dispatch{defaults::remote_io_reactor_dispatch()}
{
  // Force LibCurl global init before any reactor opens a multi handle.
  std::ignore = LibCurl::instance();

  auto const n = defaults::remote_io_num_reactors();
  KVIKIO_EXPECT(n > 0, "remote_io_num_reactors must be a positive integer", std::invalid_argument);

  auto const max_total = defaults::remote_io_max_concurrent_requests();
  // Validate the finite global budget before starting any reactor threads.
  std::ignore = reactor_concurrency_limit(max_total, n, 0);

  // Construct and publish the complete immutable reactor set before any reactor thread can report
  // a fatal error and ask the pool to wake its peers. Starting a thread while building a local
  // vector would let wakeup_all() race publication of that vector into `_reactors`.
  _reactors.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    auto const per_reactor_max = reactor_concurrency_limit(max_total, n, i);
    _reactors.emplace_back(std::make_unique<MultiPollReactor>(this, per_reactor_max));
  }

  try {
    for (auto& reactor : _reactors) {
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
      maybe_inject_failure(reactor_construction_failure_countdown);
#endif
      reactor->start();
    }
  } catch (...) {
    // Keep the complete vector stable while all successfully started threads stop. A running
    // reactor may still signal pool death and iterate the vector during this unwind.
    for (auto& reactor : _reactors) {
      reactor->stop();
    }
    throw;
  }
}

MultiReactorPool::~MultiReactorPool() noexcept
{
  // Intentionally empty. The pool is a leaked singleton, so this dtor is never invoked.
}

MultiReactorPool& MultiReactorPool::instance()
{
  // Heap-leaked singleton. The pool, its reactors, and their `std::thread`s are never destroyed.
  // Resources are cleaned on process exit.
  static MultiReactorPool* inst = new MultiReactorPool();
  return *inst;
}

void MultiReactorPool::submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers)
{
  auto const reactor_count = _reactors.size();

  // PER_PREAD: one reactor for the whole pread() call. Preserves per-CURLM connection-pool reuse.
  if (_dispatch == RemoteReactorDispatch::PER_PREAD) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    std::lock_guard<std::mutex> const submission_lock(_submission_mutex);
    _reactors[idx]->submit(std::move(transfers));
    return;
  }

  // PER_CHUNK: round-robin sub-ranges across reactors. Build the buckets before taking the global
  // gate so allocation and routing do not stall unrelated reactor inbox drains.
  std::vector<std::vector<std::unique_ptr<RemoteMultiTransfer>>> buckets(reactor_count);
  for (auto& transfer : transfers) {
    auto const idx = _next_reactor_counter.fetch_add(1, std::memory_order_relaxed) % reactor_count;
    buckets[idx].push_back(std::move(transfer));
  }

  // Hold the gate only across handoff. No reactor can drain an early bucket before every later
  // handoff either succeeds or coherently declares pool death.
  std::lock_guard<std::mutex> const submission_lock(_submission_mutex);
  for (std::size_t i = 0; i < reactor_count; ++i) {
    if (!buckets[i].empty()) { _reactors[i]->submit(std::move(buckets[i])); }
  }
}

bool MultiReactorPool::is_dead() const noexcept
{
  // This function is on a hot path, so we use atomic instead of a mutex.
  return _dead.load(std::memory_order_acquire);
}

std::exception_ptr MultiReactorPool::death_reason() const noexcept
{
  std::lock_guard<std::mutex> const lock(_death_mutex);
  return _death_reason;
}

void MultiReactorPool::signal_death(std::exception_ptr eptr) noexcept
{
  // The lock serializes _death_reason writes and keeps the _dead store in its scope so the first
  // writer wins, not the last. The store is `release`, pairing with the `acquire` in `is_dead()`.
  // The guard load below can be relaxed.
  {
    std::lock_guard<std::mutex> const lock(_death_mutex);
    // Only the first thread here updates _death_reason and wakes reactors. Later calls early-exit.
    if (_dead.load(std::memory_order_relaxed)) { return; }
    _death_reason = eptr;
    _dead.store(true, std::memory_order_release);
  }

  wakeup_all();
}

void MultiReactorPool::wakeup_all() noexcept
{
  // Including a caller's own reactor is harmless. curl_multi_wakeup is explicitly thread-safe and
  // each reactor retains a bounded poll timeout if a rare wakeup fails.
  for (auto const& reactor : _reactors) {
    reactor->wakeup();
  }
}

}  // namespace kvikio::detail
