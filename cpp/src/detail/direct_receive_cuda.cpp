/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include <kvikio/detail/direct_receive_cuda.hpp>
#include <kvikio/detail/direct_receive_stats.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {
namespace {

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
std::atomic<bool> fail_next_pre_submit_allocation{false};
#endif

[[nodiscard]] bool ranges_overlap(CUdeviceptr lhs,
                                  std::size_t lhs_size,
                                  CUdeviceptr rhs,
                                  std::size_t rhs_size) noexcept
{
  // Address arithmetic was checked before this helper is called.
  auto const lhs_end = lhs + lhs_size;
  auto const rhs_end = rhs + rhs_size;
  return lhs < rhs_end && rhs < lhs_end;
}

[[nodiscard]] CUdeviceptr checked_pointer_add(CUdeviceptr base,
                                              std::size_t offset,
                                              char const* message)
{
  constexpr auto maximum = std::numeric_limits<CUdeviceptr>::max();
  KVIKIO_EXPECT(offset <= maximum && base <= maximum - offset, message, std::invalid_argument);
  return base + offset;
}

}  // namespace

DirectReceiveCudaBatch::DirectReceiveCudaBatch(CUcontext cuda_context,
                                               CUstream stream,
                                               DirectReceiveCudaPath path)
  : _cuda_context{cuda_context}, _stream{stream}, _path{path}
{
  KVIKIO_EXPECT(_cuda_context != nullptr,
                "direct-receive CUDA batch requires a context",
                std::invalid_argument);
  KVIKIO_EXPECT(
    _path == DirectReceiveCudaPath::strict_rx || _path == DirectReceiveCudaPath::copied_stream,
    "direct-receive CUDA batch has an invalid accounting path",
    std::invalid_argument);
}

DirectReceiveCudaBatch::~DirectReceiveCudaBatch() noexcept { fail_closed_noexcept(); }

DirectReceiveCudaAddResult DirectReceiveCudaBatch::try_add(
  DirectReceiveSlotPool::Slot& slot,
  DirectReceiveReleasedSlot const& released,
  CUdeviceptr destination_base,
  std::size_t destination_extent,
  std::shared_ptr<IoEventBarrier> barrier,
  std::uintptr_t completion_cookie)
{
  KVIKIO_EXPECT(
    _state == State::collecting, "direct-receive CUDA batch is not collecting", std::logic_error);
  KVIKIO_EXPECT(slot, "direct-receive CUDA work requires a source slot", std::invalid_argument);
  KVIKIO_EXPECT(released.span_count <= direct_receive_max_spans_per_slot,
                "direct-receive descriptor exceeds its span capacity",
                std::invalid_argument);

  if (_slot_count == direct_receive_max_slots_per_cuda_batch ||
      released.span_count > direct_receive_max_copies_per_cuda_batch - _copy_count) {
    return DirectReceiveCudaAddResult::batch_full;
  }

  KVIKIO_EXPECT(barrier != nullptr,
                "direct-receive CUDA work requires a completion barrier",
                std::invalid_argument);
  KVIKIO_EXPECT(barrier->cuda_context() == _cuda_context,
                "direct-receive CUDA work belongs to a different context",
                std::invalid_argument);
  KVIKIO_EXPECT(released.raw_bytes != 0,
                "released direct-receive slot must contain raw bytes",
                std::invalid_argument);
  KVIKIO_EXPECT(released.raw_bytes <= slot.size(),
                "direct-receive raw bytes exceed the source slot",
                std::invalid_argument);
  KVIKIO_EXPECT(released.body_bytes <= released.raw_bytes,
                "direct-receive body bytes exceed raw bytes",
                std::invalid_argument);
  KVIKIO_EXPECT((released.span_count == 0) == (released.body_bytes == 0),
                "direct-receive spans and body byte count disagree",
                std::invalid_argument);
  KVIKIO_EXPECT(released.body_bytes == 0 || destination_base != 0,
                "direct-receive body requires a destination",
                std::invalid_argument);

  for (std::size_t i = 0; i < _slot_count; ++i) {
    KVIKIO_EXPECT(_slots[i].get() != slot.get(),
                  "direct-receive source slot is already in this CUDA batch",
                  std::invalid_argument);
  }

  auto const source_base = convert_void2deviceptr(slot.get());
  std::size_t body_bytes{};
  std::size_t previous_source_end{};
  std::size_t previous_destination_end{};
  for (std::size_t i = 0; i < released.span_count; ++i) {
    auto const& span = released.spans[i];
    KVIKIO_EXPECT(
      span.size != 0, "direct-receive CUDA copy cannot be empty", std::invalid_argument);
    KVIKIO_EXPECT(span.source_offset <= released.raw_bytes &&
                    span.size <= released.raw_bytes - span.source_offset,
                  "direct-receive source span exceeds raw bytes",
                  std::invalid_argument);
    KVIKIO_EXPECT(span.destination_offset <= destination_extent &&
                    span.size <= destination_extent - span.destination_offset,
                  "direct-receive destination span exceeds the requested range",
                  std::invalid_argument);
    KVIKIO_EXPECT(i == 0 || span.source_offset >= previous_source_end,
                  "direct-receive source spans overlap or are out of order",
                  std::invalid_argument);
    KVIKIO_EXPECT(i == 0 || span.destination_offset == previous_destination_end,
                  "direct-receive destination spans are not contiguous",
                  std::invalid_argument);
    KVIKIO_EXPECT(body_bytes <= std::numeric_limits<std::size_t>::max() - span.size,
                  "direct-receive body byte count overflow",
                  std::invalid_argument);

    auto const source = checked_pointer_add(
      source_base, span.source_offset, "direct-receive source address overflow");
    auto const destination = checked_pointer_add(
      destination_base, span.destination_offset, "direct-receive destination address overflow");
    // Validate each end as well as each start before using unchecked arithmetic in overlap tests.
    static_cast<void>(
      checked_pointer_add(source, span.size, "direct-receive source end address overflow"));
    static_cast<void>(checked_pointer_add(
      destination, span.size, "direct-receive destination end address overflow"));

    for (std::size_t existing = 0; existing < _copy_count; ++existing) {
      KVIKIO_EXPECT(
        !ranges_overlap(destination, span.size, _destinations[existing], _sizes[existing]),
        "direct-receive CUDA batch contains overlapping destinations",
        std::invalid_argument);
    }
    for (std::size_t prior = 0; prior < i; ++prior) {
      auto const& prior_span = released.spans[prior];
      auto const prior_destination =
        checked_pointer_add(destination_base,
                            prior_span.destination_offset,
                            "direct-receive destination address overflow");
      KVIKIO_EXPECT(!ranges_overlap(destination, span.size, prior_destination, prior_span.size),
                    "direct-receive CUDA batch contains overlapping destinations",
                    std::invalid_argument);
    }

    body_bytes += span.size;
    previous_source_end      = span.source_offset + span.size;
    previous_destination_end = span.destination_offset + span.size;
  }
  KVIKIO_EXPECT(body_bytes == released.body_bytes,
                "direct-receive span sizes do not match body bytes",
                std::invalid_argument);
  KVIKIO_EXPECT(_body_bytes <= std::numeric_limits<std::size_t>::max() - body_bytes,
                "direct-receive CUDA batch byte count overflow",
                std::invalid_argument);

  bool barrier_already_present{};
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    if (_barriers[i].get() == barrier.get()) {
      barrier_already_present = true;
      break;
    }
  }
  if (!barrier_already_present && _barrier_count == direct_receive_max_slots_per_cuda_batch) {
    return DirectReceiveCudaAddResult::batch_full;
  }

  // Every validation and capacity check is complete. Flatten only live spans, then take ownership
  // of the slot as the final non-throwing mutation.
  for (std::size_t i = 0; i < released.span_count; ++i) {
    auto const& span          = released.spans[i];
    auto const descriptor     = _copy_count + i;
    _sources[descriptor]      = source_base + span.source_offset;
    _destinations[descriptor] = destination_base + span.destination_offset;
    _sizes[descriptor]        = span.size;
  }
  if (!barrier_already_present) { _barriers[_barrier_count++] = std::move(barrier); }
  _cookies[_slot_count] = completion_cookie;
  _slots[_slot_count]   = std::move(slot);
  ++_slot_count;
  _copy_count += released.span_count;
  _body_bytes += body_bytes;
  return DirectReceiveCudaAddResult::added;
}

DirectReceiveCudaSubmission DirectReceiveCudaBatch::submit()
{
  KVIKIO_EXPECT(_state == State::collecting,
                "direct-receive CUDA batch was already submitted",
                std::logic_error);
  KVIKIO_EXPECT(
    _slot_count != 0, "cannot submit an empty direct-receive CUDA batch", std::logic_error);

  auto const submission = DirectReceiveCudaSubmission{.slots        = _slot_count,
                                                      .copies       = _copy_count,
                                                      .bytes        = _body_bytes,
                                                      .asynchronous = _copy_count != 0};
  if (_copy_count == 0) {
    recycle_slots();
    for (std::size_t i = 0; i < _barrier_count; ++i) {
      _barriers[i].reset();
    }
    _barrier_count = 0;
    _state         = State::completed;
    return submission;
  }

  PushAndPopContext current{_cuda_context};
#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
  if (fail_next_pre_submit_allocation.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc{};
  }
#endif
  // Finish every allocating preparation before the first copy can be queued.
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    _barriers[i]->prepare_event(_stream);
  }
  if (!_completion_event.has_value()) {
    _completion_event.emplace(CudaEventPool::instance().get());
  }

  bool h2d_attempted{};
  try {
    h2d_attempted = true;
    KVIKIO_CUDA_DRIVER_TRY(cudaAPI::cuda_memcpy_batch_async(
      _destinations.data(), _sources.data(), _sizes.data(), _copy_count, _stream));
    if (_path == DirectReceiveCudaPath::strict_rx) {
      direct_receive_record_strict_h2d_submission(_body_bytes);
    } else {
      direct_receive_record_copied_h2d_submission(_body_bytes);
    }
    for (std::size_t i = 0; i < _barrier_count; ++i) {
      _barriers[i]->record_prepared_event(_stream);
    }
    try {
      _completion_event->record(_stream);
    } catch (...) {
      _completion_event->abandon();
      throw;
    }
    _state = State::submitted;
    return submission;
  } catch (...) {
    auto const primary_error = std::current_exception();
    if (h2d_attempted) { recover_after_cuda_failure(primary_error); }
    std::rethrow_exception(primary_error);
  }
}

bool DirectReceiveCudaBatch::poll()
{
  if (_state == State::completed) { return true; }
  KVIKIO_EXPECT(
    _state == State::submitted, "direct-receive CUDA batch is not submitted", std::logic_error);

  try {
    PushAndPopContext current{_cuda_context};
    bool done{};
    try {
      done = _completion_event->is_done();
    } catch (...) {
      auto const primary_error = std::current_exception();
      _completion_event->abandon();
      recover_after_cuda_failure(primary_error);
    }
    if (!done) { return false; }
  } catch (...) {
    if (_state == State::submitted) {
      auto const primary_error = std::current_exception();
      _completion_event->abandon();
      mark_barriers_unknown();
      quarantine_slots();
      _state = State::failed;
      std::rethrow_exception(primary_error);
    }
    throw;
  }

  complete_successfully();
  return true;
}

void DirectReceiveCudaBatch::wait()
{
  if (_state == State::completed) { return; }
  KVIKIO_EXPECT(
    _state == State::submitted, "direct-receive CUDA batch is not submitted", std::logic_error);

  try {
    PushAndPopContext current{_cuda_context};
    try {
      _completion_event->synchronize();
    } catch (...) {
      auto const primary_error = std::current_exception();
      _completion_event->abandon();
      recover_after_cuda_failure(primary_error);
    }
  } catch (...) {
    if (_state == State::submitted) {
      auto const primary_error = std::current_exception();
      _completion_event->abandon();
      mark_barriers_unknown();
      quarantine_slots();
      _state = State::failed;
      std::rethrow_exception(primary_error);
    }
    throw;
  }
  complete_successfully();
}

std::span<std::uintptr_t const> DirectReceiveCudaBatch::finished_cookies() const
{
  KVIKIO_EXPECT(_state == State::completed || _state == State::failed,
                "direct-receive terminal cookies are not ready",
                std::logic_error);
  return {_cookies.data(), _slot_count};
}

void DirectReceiveCudaBatch::reset()
{
  KVIKIO_EXPECT(_state != State::submitted,
                "cannot reset a submitted direct-receive CUDA batch",
                std::logic_error);
  if (_state == State::collecting) { recycle_slots(); }
  clear_metadata();
  _state = State::collecting;
}

bool DirectReceiveCudaBatch::empty() const noexcept { return _slot_count == 0; }
bool DirectReceiveCudaBatch::submitted() const noexcept { return _state == State::submitted; }
bool DirectReceiveCudaBatch::completed() const noexcept { return _state == State::completed; }
bool DirectReceiveCudaBatch::failed() const noexcept { return _state == State::failed; }
std::size_t DirectReceiveCudaBatch::slot_count() const noexcept { return _slot_count; }
std::size_t DirectReceiveCudaBatch::copy_count() const noexcept { return _copy_count; }
std::size_t DirectReceiveCudaBatch::body_bytes() const noexcept { return _body_bytes; }

void DirectReceiveCudaBatch::complete_successfully() noexcept
{
  _completion_event.reset();
  recycle_slots();
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    _barriers[i].reset();
  }
  _barrier_count = 0;
  _state         = State::completed;
}

[[noreturn]] void DirectReceiveCudaBatch::recover_after_cuda_failure(
  std::exception_ptr primary_error)
{
  auto const sync_result = cudaAPI::instance().StreamSynchronize(_stream);
  if (sync_result == CUDA_SUCCESS) {
    _completion_event.reset();
    recycle_slots();
  } else {
    if (_completion_event.has_value()) { _completion_event->abandon(); }
    mark_barriers_unknown();
    quarantine_slots();
  }
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    _barriers[i].reset();
  }
  _barrier_count = 0;
  _state         = State::failed;
  std::rethrow_exception(primary_error);
}

void DirectReceiveCudaBatch::fail_closed_noexcept() noexcept
{
  if (_state == State::submitted) {
    bool synchronized{};
    try {
      PushAndPopContext current{_cuda_context};
      synchronized = cudaAPI::instance().StreamSynchronize(_stream) == CUDA_SUCCESS;
    } catch (...) {
    }
    if (synchronized) {
      complete_successfully();
    } else {
      if (_completion_event.has_value()) { _completion_event->abandon(); }
      mark_barriers_unknown();
      quarantine_slots();
      _state = State::failed;
    }
  } else if (_state == State::collecting) {
    recycle_slots();
  }
}

void DirectReceiveCudaBatch::recycle_slots() noexcept
{
  DirectReceiveSlotPool::recycle_completed({_slots.data(), _slot_count});
}

void DirectReceiveCudaBatch::quarantine_slots() noexcept
{
  for (std::size_t i = 0; i < _slot_count; ++i) {
    _slots[i].quarantine_after_failed_sync();
  }
}

void DirectReceiveCudaBatch::mark_barriers_unknown() noexcept
{
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    _barriers[i]->mark_completion_unknown();
  }
}

void DirectReceiveCudaBatch::clear_metadata() noexcept
{
  _completion_event.reset();
  for (std::size_t i = 0; i < _barrier_count; ++i) {
    _barriers[i].reset();
  }
  _slot_count    = 0;
  _copy_count    = 0;
  _barrier_count = 0;
  _body_bytes    = 0;
}

#if defined(KVIKIO_ENABLE_TEST_FAILURE_INJECTION)
void DirectReceiveCudaBatch::inject_pre_submit_allocation_failure_for_testing() noexcept
{
  fail_next_pre_submit_allocation.store(true, std::memory_order_relaxed);
}
#endif

}  // namespace kvikio::detail
