// Diagnostic schedule gate only. Original controller math and clock gates stay.
#pragma once
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wm02_callback_replay_v61 {
struct Phase {int64_t policy;int count;};
struct Clock {
 std::vector<std::vector<Phase>> rows;
 std::atomic<int64_t> current{-1};std::atomic<int> performed{0};
 explicit Clock(const std::string& path="") {
  if(path.empty())return;
  std::ifstream in(path);std::string tag;int stop=0;
  if(!(in>>tag>>stop)||tag!="WM02_CALLBACK_REPLAY_V61"||stop<2||stop>5703)
   throw std::runtime_error("INVALID_RECORDED_CALLBACK_REPLAY_HEADER");
  rows.resize(stop);
  for(int step=1;step<stop;++step){int actual,n;
   if(!(in>>actual>>n)||actual!=step||n<0||n>3)
    throw std::runtime_error("INVALID_RECORDED_CALLBACK_ROW");
   int64_t before=0;
   for(int i=0;i<n;++i){Phase p;
    if(!(in>>p.policy>>p.count)||p.policy<1||p.policy<before||p.count<1||p.count>100000)
     throw std::runtime_error("INVALID_RECORDED_CALLBACK_PHASE");
    before=p.policy;rows[step].push_back(p);
   }
  }
  std::string extra;if(in>>extra)throw std::runtime_error("EXTRA_CALLBACK_SCHEDULE_DATA");
 }
 bool active(int64_t step)const{return step>0&&step<int64_t(rows.size());}
 int done(int64_t step)const{return current.load()==step?performed.load():0;}
 int target(int64_t step)const{int n=0;for(auto p:rows[step])n+=p.count;return n;}
 bool control_allowed(int64_t step,int64_t published_policy)const{
  if(!active(step))return true;
  int count=done(step),end=0;
  for(auto p:rows[step]){end+=p.count;if(count<end)return published_policy==p.policy;}
  return false;
 }
 // Called after the unchanged SetCommand,under its existing simulation mutex.
 void control_complete(int64_t step){
  if(!active(step))return;
  if(current.load()!=step){performed.store(0);current.store(step);}
  performed.fetch_add(1);
 }
 bool policy_allowed(int64_t step,int64_t policy)const{
  if(!active(step))return true;
  int preceding=0;
  for(auto p:rows[step]){
   if(p.policy==policy)return done(step)>=preceding;
   if(p.policy<policy)preceding+=p.count;
  }
  return false;
 }
 // Conjoined with the existing applied-policy/control clock,never replaces it.
 bool integration_ready(int64_t step)const{
  return !active(step)||done(step)==target(step);
 }
};
inline int64_t tick(double time){return std::llround(time*1e9)/2000000;}
inline Clock& clock(){static Clock value(std::getenv("WM02_CALLBACK_REPLAY_V61")?
 std::getenv("WM02_CALLBACK_REPLAY_V61"):"");return value;}
}
