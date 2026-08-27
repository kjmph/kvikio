/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/concurrent_request_limiter.hpp>
#include <kvikio/detail/direct_receive.hpp>
#include <kvikio/detail/direct_receive_cuda.hpp>
#include <kvikio/detail/direct_receive_slot_pool.hpp>
#include <kvikio/detail/http_retry.hpp>
#include <kvikio/detail/io_event_barrier.hpp>
#include <kvikio/detail/observation_recorder.hpp>
#include <kvikio/detail/remote_callback.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {

class MultiReactorPool;  // Forward declaration, because reactors needs to hold a back-pointer to
                         // the pool.

/**
 * @brief Compute one reactor's exact private share of the process-wide concurrency budget.
 *
 * A zero total means unlimited. A finite total must be at least the reactor count so every reactor
 * can admit work. Any remainder is assigned one request at a time to the lowest reactor indexes;
 * summing all returned shares therefore equals @p max_total exactly.
 *
 * @param max_total Process-wide maximum concurrent request count, or zero for unlimited.
 * @param num_reactors Number of reactors sharing the budget.
 * @param reactor_index Zero-based index of the reactor whose share is requested.
 * @return This reactor's finite private share, or `std::nullopt` for unlimited concurrency.
 * @throws std::invalid_argument if the reactor count/index is invalid or a finite budget is smaller
 * than the reactor count.
 */
[[nodiscard]] std::optional<std::size_t> reactor_concurrency_limit(std::size_t max_total,
                                                                   std::size_t num_reactors,
                                                                   std::size_t reactor_index);

/**
 * @brief Given the max concurrent request cap for a reactor, derive the size of the libcurl
 * connection cache (`CURLMOPT_MAXCONNECTS`).
 *
 * @param max_concurrent_requests This reactor's private share of the total concurrent-request
 * budget (the global cap divided across reactors). `std::nullopt` means unlimited.
 * @return The value to pass to `CURLMOPT_MAXCONNECTS`. `std::nullopt` if @p max_concurrent_requests
 * is `std::nullopt` (unlimited concurrency).
 */
[[nodiscard]] std::optional<long> connection_cache_size(
  std::optional<std::size_t> max_concurrent_requests) noexcept;

/**
 * @brief Collects results from N sub-range transfers and resolves one top-level future once every
 * sub-range has either succeeded or failed.
 *
 * Every sub-range transfer belonging to a single `RemoteHandle::pread()` call holds a
 * `std::shared_ptr<RemoteMultiAggregateContext>`. As completions arrive on the reactor threads
 * (potentially in parallel when `KVIKIO_REMOTE_IO_NUM_REACTORS > 1`), each one calls
 * `on_subrange_complete()` or `on_subrange_failed()`. The thread that decrements `_subranges_left`
 * to zero fulfills `_promise`, with the accumulated byte total on success, or with the first
 * captured exception on failure.
 */
class RemoteMultiAggregateContext {
 public:
  /**
   * @brief Construct an aggregate that expects exactly `num_subranges` completion events.
   *
   * @param num_subranges Number of sub-range transfers the caller has split the read into.
   */
  explicit RemoteMultiAggregateContext(std::size_t num_subranges);

  /**
   * @brief Per-pread event barrier for the device-buffer path.
   */
  std::shared_ptr<IoEventBarrier> io_event_barrier;

  /**
   * @brief Records the logical operation these sub-ranges make up. Null when nobody is observing.
   */
  std::shared_ptr<LogicalObservationRecorder> recorder;

  /**
   * @brief Report that one sub-range transfer succeeded.
   *
   * @param bytes Number of bytes the sub-range delivered.
   */
  void on_subrange_complete(std::size_t bytes);

  /**
   * @brief Report that one sub-range transfer failed. The first exception captured wins.
   *
   * @param eptr The exception describing the failure.
   */
  void on_subrange_failed(std::exception_ptr eptr);

  /**
   * @brief Obtain the future the caller will observe. Must be called exactly once, before any
   * sub-range is submitted to the pool.
   */
  std::future<std::size_t> get_future();

 private:
  std::atomic<std::size_t> _subranges_left;
  std::atomic<std::size_t> _total_bytes{0};
  std::mutex _exception_mutex;
  std::exception_ptr _first_exception;
  std::promise<std::size_t> _promise;
};

/**
 * @brief RAII guard that keeps one libcurl easy handle attached to a multi handle.
 *
 * Set by the reactor right after a successful `curl_multi_add_handle`. Its destructor calls
 * `curl_multi_remove_handle`, so the handle is detached when the owning `RemoteMultiTransfer` is
 * destroyed. A default-constructed or moved-from guard is unset and does nothing on destruction.
 *
 * @note Must be destroyed on the reactor I/O thread that set it, because `CURLM*` is not
 * thread-safe. It is a `RemoteMultiTransfer` member declared after `curl`, so it detaches the
 * handle before `CurlHandle` returns it to the LibCurl pool.
 */
class CurlMultiAttachment {
 public:
  /**
   * @brief Construct an unset guard that holds no attachment.
   */
  CurlMultiAttachment() noexcept = default;

  /**
   * @brief Set a guard for an easy handle already attached to `multi`.
   *
   * @param multi The multi handle the easy handle was added to.
   * @param easy The easy handle to remove on destruction.
   */
  CurlMultiAttachment(CURLM* multi, CURL* easy) noexcept;

  ~CurlMultiAttachment();

  /**
   * @brief Explicitly detach the easy handle now instead of at destruction.
   */
  void reset() noexcept;

  // Move-only.
  CurlMultiAttachment(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment& operator=(CurlMultiAttachment&& o) noexcept;
  CurlMultiAttachment(CurlMultiAttachment const&)            = delete;
  CurlMultiAttachment& operator=(CurlMultiAttachment const&) = delete;

 private:
  CURLM* _multi{nullptr};
  CURL* _easy{nullptr};
};

/**
 * @brief Per-transfer state owned by a `MultiPollReactor` between submission and completion.
 *
 * One `RemoteMultiTransfer` corresponds to one libcurl easy handle, which corresponds to one HTTP
 * range request. Sub-ranges of the same `pread()` share the same `aggregate`. The `curl` member is
 * held by `std::unique_ptr` because `CurlHandle` is intentionally non-movable.
 */
struct RemoteMultiTransfer {
  std::unique_ptr<CurlHandle> curl;

  // Detaches `curl`'s easy handle from the multi handle on destruction.
  CurlMultiAttachment attachment;

  CallbackContext ctx;
  std::shared_ptr<RemoteMultiAggregateContext> aggregate;

  // Concurrency slot held from stage (1) admission until this transfer is destroyed after
  // completion or failure. Empty while the transfer waits in the inbox. Destroying the transfer
  // returns the slot to the reactor's limiter.
  ConcurrentRequestLimiter::Slot slot;

  // Device-path fields. All zeroed/null for host transfers.
  bool is_device{false};
  CUcontext device_ctx{nullptr};
  void* device_dst{nullptr};
  CudaPinnedBounceBufferPool::Buffer buffer{nullptr, nullptr, 0};

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  /**
   * @brief Reactor-owned state for one caller-owned receive transfer.
   *
   * Network completion and destination completion are deliberately separate. `network_done`
   * releases the easy-handle attachment and request-limiter slot. GPU reads remain alive until
   * every pinned source slot has reached a terminal CUDA batch; host reads finish their synchronous
   * placement before returning from the reactor iteration.
   */
  struct DirectReceiveState {
    ~DirectReceiveState();

    std::size_t file_offset{};
    bool fallback_allowed{};
    bool strict_attempt{true};
    bool strict_activation_recorded{};
    bool uses_pinned_slots{};
    bool waiting_for_slot{};
    bool network_done{};
    bool failure_recorded{};
    bool host_buffer_is_direct_tail{};
    CURLcode network_result{CURLE_OK};
    std::size_t host_destination_offset{};
    std::size_t host_direct_bytes{};
    std::size_t host_staged_bytes{};
    std::size_t consumer_slots_outstanding{};
    std::exception_ptr consumer_failure;
    // Lazily allocated only after network admission and not retained across retry backoff, keeping
    // ordinary host scratch bounded by active request concurrency.
    std::vector<std::byte> host_framing_buffer;
    std::unique_ptr<CurlDirectReceiveState> callbacks;
    DirectReceiveSlotPool::Slot receive_slot;
  };

  std::unique_ptr<DirectReceiveState> direct_receive;
#endif

  // Guards the aggregate promise against duplicate terminal callbacks while transfer ownership
  // moves through exceptional reactor paths.
  bool aggregate_terminal_reported{false};

  // Retry bookkeeping. Number of attempts that have finished.
  std::size_t attempt{0};

  // Earliest time this transfer may be admitted. Used to space out retries.
  // The default is the clock epoch, which is always in the past, so a freshly submitted transfer is
  // admitted immediately.
  std::chrono::steady_clock::time_point ready_at{};

  std::shared_ptr<HttpRetryPolicy const> retry_policy;

  /**
   * @brief Recycles `buffer` to the bounce-buffer cache if it was not already moved out (due to
   * failure paths).
   **/
  ~RemoteMultiTransfer();
};

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
/**
 * @brief Make the next reactor submission throw after this many successful inbox insertions.
 *
 * The hook resets itself after firing and is compiled only into test builds.
 */
void inject_multi_poll_submission_failure_after_for_testing(
  std::size_t successful_insertions) noexcept;

/**
 * @brief Make the next reactor admission throw before reserving its in-flight map node.
 *
 * The hook resets itself after firing and is compiled only into test builds.
 */
void inject_multi_poll_admission_failure_after_for_testing(
  std::size_t successful_admissions) noexcept;

/**
 * @brief Make reactor-pool construction fail after this many reactors are started.
 *
 * The hook resets itself after firing and is compiled only into test builds.
 */
void inject_multi_poll_reactor_construction_failure_after_for_testing(
  std::size_t successful_reactors) noexcept;
#endif

/**
 * @brief One reactor has one `CURLM*`, one I/O thread, one submit queue, one in-flight map.
 *
 * `CURLM*` is not thread-safe. All multi-side calls (`curl_multi_add_handle`, `curl_multi_perform`,
 * `curl_multi_info_read`, `curl_multi_remove_handle`, `curl_multi_poll`) happen on `_io_thread`.
 * The only cross-thread libcurl call is `curl_multi_wakeup()`, used by `submit()` to nudge the
 * reactor out of its poll.
 *
 * @note Instances are intentionally never destroyed after successful singleton construction.
 * Their destructor exists so a partially constructed `MultiReactorPool` can clean up reactors
 * after startup fails.
 */
class MultiPollReactor {
 public:
  /**
   * @brief Construct a reactor owned by the given pool.
   *
   * @param pool Non-owning back-pointer to the pool that owns this reactor. Used to observe and
   * propagate pool-wide death state. The pool must outlive the reactor, which is guaranteed because
   * the pool is a leaked singleton that owns this reactor by `unique_ptr`.
   * @param max_concurrent_requests This reactor's private share of the total concurrent-request
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited. Each reactor
   * enforces its own share against its own inbox.
   */
  MultiPollReactor(MultiReactorPool* pool, std::optional<std::size_t> max_concurrent_requests);
  ~MultiPollReactor() noexcept;
  MultiPollReactor(MultiPollReactor const&)            = delete;
  MultiPollReactor& operator=(MultiPollReactor const&) = delete;
  MultiPollReactor(MultiPollReactor&&)                 = delete;
  MultiPollReactor& operator=(MultiPollReactor&&)      = delete;

  /**
   * @brief Hand off a batch of prepared transfers to this reactor. Thread-safe.
   *
   * The reactor picks the transfers up on its next loop iteration. The caller must have already
   * obtained the aggregate future via `aggregate->get_future()` before calling this, because once
   * the transfers are in the queue the reactor may complete them (and the promise) at any time. If
   * the pool has already declared death, every transfer in the batch is failed immediately with
   * the recorded death reason and never enters the inbox.
   *
   * Once any transfer is visible to a reactor this call never propagates an exception. A queue
   * allocation failure declares the pool dead and resolves every transfer through its aggregate,
   * preserving the caller's future and device-I/O fence contract.
   *
   * @param transfers Per-transfer state, ownership transferred to the reactor.
   */
  void submit(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers) noexcept;

  /**
   * @brief Wake up the reactor out of its `curl_multi_poll()` wait. Thread-safe.
   *
   * This method calls `curl_multi_wakeup()`. If it fails (which is rare) the reactor still wakes on
   * its bounded poll timeout. Used by `MultiReactorPool::signal_death` to make every reactor notice
   * pool death promptly rather than waiting for the timeout.
   */
  void wakeup() noexcept;

 private:
  friend class MultiReactorPool;

  /** @brief Start the reactor thread after the pool has published its complete reactor set. */
  void start();

  /** @brief Request shutdown and join the reactor thread, if it was started. */
  void stop() noexcept;

  /**
   * @brief Set this reactor's libcurl connection cache (`CURLMOPT_MAXCONNECTS`).
   *
   * By default libcurl sets `CURLMOPT_MAXCONNECTS` to 4 x the number of easy handles attached to a
   * multi handle. This is recomputed on every transition, and a transient dip in concurrency will
   * cause libcurl to evict warm, reusable connections, and cause unnecessary TCP/TLS handshake.
   * Here we pin `CURLMOPT_MAXCONNECTS` to a fixed size.
   *
   * @param max_concurrent_requests This reactor's private share of the total concurrent-request
   * budget (the global cap divided across reactors). `std::nullopt` means unlimited.
   *
   * @exception std::runtime_error if `curl_multi_setopt` fails.
   */
  void set_connection_cache_size(std::optional<std::size_t> max_concurrent_requests) const;

  void io_thread_main();

  /**
   * @brief Fail every transfer this reactor is responsible for and exit the loop.
   *
   * Called from the I/O thread on its way out, either because this reactor caught an exception or
   * because another reactor signaled pool death. Drains the inbox, removes each in-flight easy
   * handle from the multi handle, and resolves each transfer's aggregate with the given exception.
   */
  void fail_all_pending(std::exception_ptr eptr);

  /**
   * @brief Requeue a failed transfer in `_pending` so it can be attempted again.
   *
   * @param transfer The transfer to requeue. Ownership moves into `_pending`.
   * @param ready_at Earliest time the transfer may be admitted again.
   */
  void requeue_for_retry(std::unique_ptr<RemoteMultiTransfer> transfer,
                         std::chrono::steady_clock::time_point ready_at) noexcept;

#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  struct DirectReceiveCudaWork {
    CUcontext cuda_context{};
    CUstream stream{};
    DirectReceiveCudaPath path{DirectReceiveCudaPath::strict_rx};
    RemoteMultiAggregateContext* aggregate{};
    std::unique_ptr<DirectReceiveCudaBatch> batch;
    std::exception_ptr submission_failure;
    std::array<RemoteMultiTransfer*, direct_receive_max_slots_per_cuda_batch> owners{};
    std::size_t owner_count{};
  };

  [[nodiscard]] bool try_install_direct_receive_slot(RemoteMultiTransfer& transfer,
                                                     bool resume_transfer);
  void collect_direct_receive_slot(RemoteMultiTransfer& transfer);
  void resume_waiting_direct_receive_transfers();
  void submit_direct_receive_batch(DirectReceiveCudaWork& work);
  void submit_collecting_direct_receive_batches();
  [[nodiscard]] bool reap_direct_receive_batches();
  void fail_direct_receive_after_consumer(RemoteMultiTransfer& transfer,
                                          std::exception_ptr failure) noexcept;
  void finish_direct_receive_transfers(
    std::optional<std::chrono::steady_clock::time_point>& earliest_ready_at);
  void requeue_direct_receive(std::unique_ptr<RemoteMultiTransfer> transfer,
                              bool strict_attempt,
                              std::chrono::steady_clock::time_point ready_at) noexcept;
  void record_direct_receive_failure(RemoteMultiTransfer& transfer, bool protocol_failure) noexcept;
  [[nodiscard]] bool direct_receive_pool_is_permanently_exhausted() const;
  static void CUDA_CB cuda_completion_wakeup(void* user_data) noexcept;
#endif

  MultiReactorPool* _pool;
  ConcurrentRequestLimiter _request_limiter;
  CURLM* _curl_multi{nullptr};
  std::thread _io_thread;
  std::atomic<bool> _stop{false};
  std::mutex _submit_mutex;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _inbox;
  std::deque<std::unique_ptr<RemoteMultiTransfer>> _pending;
  // Direct-receive entries remain here after their easy handle is detached while released receive
  // slots drain through CUDA. Their stable heap addresses are the CUDA batch's owner cookies.
  std::unordered_map<CURL*, std::unique_ptr<RemoteMultiTransfer>> _in_flight;
#if defined(CURL_HAS_RECV_BUFFER_CALLBACKS) && defined(CURL_HAS_KTLS_DIRECT_RX)
  std::deque<DirectReceiveCudaWork> _direct_receive_cuda_work;
  std::atomic<bool> _cuda_completion_hint{false};
  std::atomic<bool> _cuda_wakeup_failed{false};
  bool _cuda_host_wakeup_enabled{true};
#endif
};

/**
 * @brief Process-wide pool that owns N reactors and dispatches sub-range transfers to them.
 *
 * Accessed via the leaked-pointer singleton `instance()`. Both `num_reactors` and the dispatch
 * mode are captured once at first use from `kvikio::defaults` and remain immutable for the process
 * lifetime: switching either requires restarting with different `KVIKIO_REMOTE_IO_NUM_REACTORS` /
 * `KVIKIO_REMOTE_IO_REACTOR_DISPATCH` env vars.
 *
 * Dispatch rules (with `N = _reactors.size()`):
 *  - `PER_CHUNK` (default): each sub-range is routed independently via a round-robin atomic
 *    counter. Maximizes load distribution. May cause sub-ranges of the same file to use distinct
 *    TCP/TLS connections.
 *  - `PER_PREAD`: all sub-ranges of one `submit_pread()` call land on the same reactor (round-robin
 *    per call). Preserves per-`CURLM` connection-pool reuse.
 */
class MultiReactorPool {
 public:
  /**
   * @brief Get the process-wide pool, creating it (and its reactor threads) on first use.
   *
   * @note The returned reference points to a heap-allocated singleton that is intentionally never
   * destroyed, mirroring the leak convention used by `BounceBufferPool` and
   * `StreamCachePerThreadAndContext`. This avoids static-destruction-order coupling between the
   * pool, `LibCurl`, the reactor threads, and (future) CUDA teardown.
   */
  static MultiReactorPool& instance();

  MultiReactorPool(MultiReactorPool const&)            = delete;
  MultiReactorPool& operator=(MultiReactorPool const&) = delete;
  MultiReactorPool(MultiReactorPool&&)                 = delete;
  MultiReactorPool& operator=(MultiReactorPool&&)      = delete;

  /**
   * @brief Submit all sub-range transfers belonging to one `RemoteHandle::pread()` call.
   *
   * Routes each transfer to a reactor according to the captured dispatch policy. The caller must
   * have already obtained the aggregate future from the shared `RemoteMultiAggregateContext`
   * before invoking this, because as soon as the pool returns the reactors may have already
   * started completing the transfers.
   *
   * @param transfers The sub-range transfers, ownership transferred to the pool.
   */
  void submit_pread(std::vector<std::unique_ptr<RemoteMultiTransfer>> transfers);

  /**
   * @brief Whether the pool has been marked dead by a fatal reactor or ownership-handoff error.
   *
   * Once dead, the pool stays dead for the rest of the process lifetime. All in-flight and
   * subsequently submitted transfers fail with the recorded death reason.
   */
  [[nodiscard]] bool is_dead() const noexcept;

  /**
   * @brief Get the exception that caused pool death, or a null `exception_ptr` if alive.
   *
   * Safe to call from any thread. Returns the same value once `is_dead()` returns `true`.
   */
  [[nodiscard]] std::exception_ptr death_reason() const noexcept;

  /**
   * @brief Mark the pool as dead with the given exception as the cause, then wake every reactor so
   * each notices the death state promptly. Thread-safe. Only the first call wins. All subsequent
   * calls are silently ignored.
   *
   * @param eptr The exception that causes pool death. Will be propagated to every in-flight and
   * subsequently submitted transfer via `RemoteMultiAggregateContext::on_subrange_failed`.
   */
  void signal_death(std::exception_ptr eptr) noexcept;

  /**
   * @brief Wake every reactor so it promptly observes a process-wide state change.
   */
  void wakeup_all() noexcept;

 private:
  friend class MultiPollReactor;

  MultiReactorPool();
  ~MultiReactorPool() noexcept;

  RemoteReactorDispatch _dispatch;
  // Makes the handoff of all reactor buckets from one pread atomic with respect to inbox draining.
  // Without this gate, a PER_CHUNK reactor could begin I/O before a later reactor's queue
  // allocation fails, leaving the caller without one coherent future/fence lifecycle.
  std::mutex _submission_mutex;
  // Round-robin counter. Incremented per pread (PER_PREAD) or per chunk (PER_CHUNK).
  std::atomic<std::size_t> _next_reactor_counter{0};
  std::atomic<bool> _dead{false};
  std::mutex mutable _death_mutex;  // Protects writes to `_death_reason`.
  std::exception_ptr _death_reason;
  // Keep reactor ownership last so an explicitly destroyed pool stops and joins every reactor
  // before destroying the synchronization and death-state members their threads inspect.
  std::vector<std::unique_ptr<MultiPollReactor>> _reactors;
};

}  // namespace kvikio::detail
