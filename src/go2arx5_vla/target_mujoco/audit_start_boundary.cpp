// CPU boundary probe: exits before the first integration; never calls mj_step.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <execinfo.h>
static void record(const char* kind,const mjModel* m,const mjData* d) {
  std::ofstream f(std::getenv("PI0288_CPU_BOUNDARY_TRACE"),std::ios::app);
  if(!f)_exit(76);
  f<<std::setprecision(17)<<"{\"kind\":\""<<kind<<"\",\"data_address\":\""<<d<<"\",\"time\":"<<d->time<<",\"qpos\":[";
  for(int i=0;i<m->nq;++i){if(i)f<<',';f<<d->qpos[i];}
  f<<"]}\n";
}
extern "C" void mj_resetDataKeyframe(const mjModel* m,mjData* d,int key) {
  using Fn=void(*)(const mjModel*,mjData*,int);
  auto real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"mj_resetDataKeyframe"));
  if(!real)_exit(76);real(m,d,key);record("after_key_reset",m,d);
}
extern "C" void mj_step(const mjModel* m,mjData* d) {
  record("before_first_step_NOT_EXECUTED",m,d);
  void* stack[24];int n=backtrace(stack,24);backtrace_symbols_fd(stack,n,2);_exit(75);
}
