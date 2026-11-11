// Single native sensor producer. Neither publication nor a read attempt waits.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
namespace wm02_robot_state_snapshot {
static_assert(std::atomic<float>::is_always_lock_free && std::atomic<uint64_t>::is_always_lock_free
              && std::atomic<int64_t>::is_always_lock_free,
              "Native sensor handoff requires lock-free atomics");
struct Frame { std::array<float,49> values{}; int64_t sample_ns=-1; };
class Snapshot {
 std::atomic<uint64_t> generation_{0};
 std::array<std::atomic<float>,49> values_{};
 std::atomic<int64_t> sample_ns_{-1};
public:
 void publish(const Frame& frame) {
  // Payload atomics avoid a C++ data race even during a rejected mixed read.
  generation_.fetch_add(1);
  for(unsigned i=0;i<49;++i)values_[i].store(frame.values[i]);
  sample_ns_.store(frame.sample_ns);generation_.fetch_add(1);
 }
 bool try_read(Frame& frame,int64_t expected_ns)const {
  auto before=generation_.load();if(!before || (before&1))return false;
  for(unsigned i=0;i<49;++i)frame.values[i]=values_[i].load();
  frame.sample_ns=sample_ns_.load();auto after=generation_.load();
  return before==after && !(after&1) && frame.sample_ns==expected_ns;
 }
};
inline Snapshot& snapshot() { static Snapshot value; return value; }
}
