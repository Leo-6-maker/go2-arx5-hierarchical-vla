#pragma once
#include "native_home.hpp"
#include <chrono>

// PI0289 V1: delivered MPC initialState.arm, not a posture sweep.
inline void PI0289ResetStartup(const mjModel* m, mjData* d) {
  PI0288ResetHome(m,d);
  for (const char* name : {"x5_joint2", "x5_joint3"}) {
    int j=mj_name2id(m,mjOBJ_JOINT,name);
    if(j<0 || m->jnt_type[j]!=mjJNT_HINGE || !m->jnt_limited[j] ||
       m->jnt_range[2*j]>=.9 || m->jnt_range[2*j+1]<=.9)
      throw std::runtime_error("PI0289 invalid named startup joint");
    d->qpos[m->jnt_qposadr[j]]=.9;
  }
  mj_forward(m,d);
  if(const char* path=std::getenv("PI0289_INITIALIZATION_TRACE")) {
    std::ofstream f(path,std::ios::app);
    if(!f)throw std::runtime_error("PI0289 initialization trace unavailable");
    f<<std::setprecision(17)<<"{\"kind\":\"actual_startup_V1\",\"time\":"<<d->time<<",\"data_pointer\":"<<reinterpret_cast<uintptr_t>(d)<<",\"qpos\":[";
    for(int i=0;i<m->nq;++i){if(i)f<<',';f<<d->qpos[i];}
    f<<"],\"qvel\":[";
    for(int i=0;i<m->nv;++i){if(i)f<<',';f<<d->qvel[i];}
    f<<"],\"ctrl\":[";
    for(int i=0;i<m->nu;++i){if(i)f<<',';f<<d->ctrl[i];}
    f<<"]}\n";
  }
}

template<class Command>
inline void PI0289ControlTrace(const mjData* d,const std::string& fsm,
                              const std::string& mode,bool running,const Command& command) {
  const char* path=std::getenv("PI0289_CONTROL_TRACE");if(!path)return;
  static std::ofstream f(path);if(!f)std::abort();
  static long calls=0;static std::string previous;
  std::string state=fsm+":"+mode+":"+std::to_string(running);
  if(++calls%5 && state==previous)return;
  previous=state;
  double mono=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  f<<std::setprecision(17)<<"{\"sim_time_s\":"<<d->time<<",\"monotonic_s\":"<<mono<<",\"control_calls\":"<<calls<<",\"fsm\":\""<<fsm<<"\",\"arm_mode\":\""<<mode<<"\",\"running\":"<<(running?"true":"false");
  auto array=[&](const char* name,const auto& v){f<<",\""<<name<<"\":[";for(int i=0;i<18;++i){if(i)f<<',';f<<v[i];}f<<']';};
  array("q_target",command.q);array("kp",command.kp);array("kd",command.kd);array("tau",command.tau);
  f<<",\"ctrl\":[";for(int i=0;i<18;++i){if(i)f<<',';f<<d->ctrl[i];}f<<"]}\n";f.flush();
}
