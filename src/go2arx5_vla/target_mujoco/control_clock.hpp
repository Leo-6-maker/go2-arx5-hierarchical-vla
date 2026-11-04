// Bind frozen native callback periods to physical time. Readers never wait.
#pragma once
#include <atomic>
#include <cstdint>
#include <cmath>
namespace wm02_control_clock {
struct Clock {
 std::atomic<int64_t> control_ns{-1},control_period_ns{0},requested_policy{-1},completed_policy{-1},policy_period_ns{0};
 static int64_t ns(double time){return std::llround(time*1e9);}
 void reset(){control_ns=-1;requested_policy=-1;completed_policy=-1;}
 void control(double time,int64_t period){control_period_ns.store(period);control_ns.store(ns(time));}
 bool policy_due(double time,int64_t period){
  if(period<=0)return false;policy_period_ns.store(period);const auto bucket=ns(time)/period;
  auto last=requested_policy.load();if(bucket<=last)return false;requested_policy.store(bucket);return true;
 }
 void policy_complete(double time,int64_t period){completed_policy.store(ns(time)/period);}
 bool ready(double time)const{
  const auto cp=control_period_ns.load(),pp=policy_period_ns.load(),c=control_ns.load(),p=completed_policy.load();
  return cp>0&&pp>0&&c>=0&&p>=0&&c/cp>=ns(time)/cp&&p>=ns(time)/pp;
 }
};
inline Clock& clock(){static Clock value;return value;}
}
