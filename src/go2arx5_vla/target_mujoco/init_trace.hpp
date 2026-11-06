// Observability only: no locks, waits, file IO or command changes on emit paths.
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <initializer_list>
#include <utility>
extern "C" uint64_t WM02InitBegin(double);
extern "C" uint64_t WM02InitCurrent();
extern "C" void WM02InitEmit(uint32_t,double,uint64_t,const float*,uint32_t);
extern "C" void WM02InitNumeric(uint32_t,double,uint64_t,const double*,uint32_t);
extern "C" void WM02InitPublish(uint32_t,uint64_t);
extern "C" void WM02InitConsume(uint32_t);
extern "C" void WM02InitApplied();
namespace wm02_init {
enum Kind:uint32_t {OBS=1,INFERENCE_BEGIN=2,INPUT=3,HISTORY=4,INFERENCE_DONE=5,PUBLISH=6,CONSUME=7,APPLY=8,PD_TARGET=9,PD_STATE_CTRL=10,STEP_BEFORE=11,STEP_AFTER=12,STOP=13,REFUSED=14,INFERENCE_SKIPPED=15,MODE=16,KEY_RECEIVED=17,KEY_APPLIED=18};
inline void emit(Kind k,double t,uint64_t v,const std::vector<float>&x={}) {WM02InitEmit(k,t,v,x.data(),x.size());}
inline void vectors(Kind k,double t,uint64_t v,std::initializer_list<const std::vector<float>*> xs) {
 std::array<double,96>b{};uint32_t n=0;for(auto x:xs)for(float f:*x)if(n<b.size())b[n++]=f;WM02InitNumeric(k,t,v,b.data(),n);
}
inline void numbers(Kind k,double t,uint64_t v,std::initializer_list<std::pair<const double*,int>> xs) {
 std::array<double,96>b{};uint32_t n=0;for(auto x:xs)for(int j=0;j<x.second;++j)if(n<b.size())b[n++]=x.first[j];WM02InitNumeric(k,t,v,b.data(),n);
}
template<class Queue>bool pop(Queue&q,std::vector<float>&v,uint32_t channel) {
 bool ok=q.try_pop(v);if(ok)WM02InitConsume(channel);return ok;
}
}
