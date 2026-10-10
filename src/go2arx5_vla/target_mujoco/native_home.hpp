#pragma once
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

// Shared by actual native creation/reset and pre-execution CPU audit.
inline void PI0288ResetHome(const mjModel* m,mjData* d) {
  int key=mj_name2id(m,mjOBJ_KEY,"home");
  if(key<0)throw std::runtime_error("PI0288 missing delivered home");
  mj_resetDataKeyframe(m,d,key);
  for(int i=0;i<m->nq;++i)
    if(!std::isfinite(d->qpos[i]))throw std::runtime_error("PI0288 nonfinite home");
  for(int j=0;j<m->njnt;++j) {
    if(!m->jnt_limited[j] || (m->jnt_type[j]!=mjJNT_HINGE && m->jnt_type[j]!=mjJNT_SLIDE))continue;
    double q=d->qpos[m->jnt_qposadr[j]];
    if(q<m->jnt_range[2*j] || q>m->jnt_range[2*j+1])
      throw std::runtime_error("PI0288 home violates raw joint limit");
  }
  mj_forward(m,d);
  if(const char* path=std::getenv("PI0288_INITIALIZATION_TRACE")) {
    std::ofstream f(path,std::ios::app);
    if(!f)throw std::runtime_error("PI0288 initialization trace unavailable");
    f<<std::setprecision(17)<<"{\"kind\":\"actual_native_home\",\"time\":"<<d->time<<",\"qpos\":[";
    for(int i=0;i<m->nq;++i){if(i)f<<',';f<<d->qpos[i];}
    f<<"],\"qvel\":[";
    for(int i=0;i<m->nv;++i){if(i)f<<',';f<<d->qvel[i];}
    f<<"],\"ctrl\":[";
    for(int i=0;i<m->nu;++i){if(i)f<<',';f<<d->ctrl[i];}
    f<<"],\"ncon\":"<<d->ncon<<"}\n";
  }
}
