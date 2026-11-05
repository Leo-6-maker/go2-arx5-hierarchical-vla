// V2: completing inference does not establish that PD applied its command.
// Publication/consumption use the existing short pair handoff; PD never waits.
#pragma once
#include "control_clock.hpp"
namespace wm02_applied_control_clock {
struct Clock : wm02_control_clock::Clock {
 std::atomic<int64_t> published_policy{-1},applied_policy{-1};
 void reset(){wm02_control_clock::Clock::reset();published_policy=-1;applied_policy=-1;}
 // Called inside the existing publication lock, before the q/dq pushes.
 void published(double time,int64_t period){published_policy.store(ns(time)/period);}
 // Called only following successful consumption of the actual coherent pair.
 // Physics cannot advance to the next policy bucket until this acknowledgement.
 void applied(){applied_policy.store(published_policy.load());}
 bool ready(double time)const{
  const auto pp=policy_period_ns.load(),a=applied_policy.load();
  return pp>0&&a>=0&&a>=ns(time)/pp&&wm02_control_clock::Clock::ready(time);
 }
};
inline Clock& clock(){static Clock value;return value;}
}
