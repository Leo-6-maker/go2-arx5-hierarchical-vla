// Read-only observer around the existing MuJoCo integration call.
// Original mjData is never modified here. FK is refreshed on a private copy.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <signal.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>

namespace {
double clock_seconds(bool wall) {
  if (wall) return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void quoted(std::ostream& f, const char* value) {
  f << '"';
  for (const char* p=value ? value : ""; *p; ++p) {
    if (*p=='"' || *p=='\\') f << '\\';
    if (static_cast<unsigned char>(*p)<32) f << '?'; else f << *p;
  }
  f << '"';
}
void array(std::ostream& f, const mjtNum* v, int n) {
  f << '['; for(int i=0;i<n;++i) { if(i) f<<','; f<<v[i]; } f<<']';
}
std::ofstream trace;
mjData* snapshot=nullptr;
std::vector<double> violation_duration;
long count=0;
long cap=0;
double next_sample=0;
bool initialized=false;
bool safety_stopped=false;
const mjModel* observed_model=nullptr;
void initialize(const mjModel* m) {
  const char* path=std::getenv("PI0288_PHYSICS_TRACE");
  if (!path) return;
  trace.open(path, std::ios::out|std::ios::app);
  if (!trace) { std::abort(); }
  trace<<std::setprecision(17);
  const char* value=std::getenv("PI0288_PHYSICS_CAP");
  if(!value || (cap=std::strtol(value,nullptr,10))<=0) std::abort();
  snapshot=mj_makeData(m);
  observed_model=m;
  if(mjcb_control || mjcb_sensor || mjcb_passive || mjcb_act_dyn || mjcb_act_gain || mjcb_act_bias)std::abort();
  violation_duration.assign(m->njnt,0);
  trace<<"{\"kind\":\"model\",\"nq\":"<<m->nq<<",\"nv\":"<<m->nv<<",\"nu\":"<<m->nu
       <<",\"timestep\":"<<m->opt.timestep<<",\"version\":";quoted(trace,mj_versionString());
  trace<<",\"joints\":[";
  for(int i=0;i<m->njnt;++i) {
    if(i) trace<<',';
    trace<<"{\"name\":";quoted(trace,mj_id2name(m,mjOBJ_JOINT,i));
    trace<<",\"type\":"<<m->jnt_type[i]<<",\"limited\":"<<int(m->jnt_limited[i])
         <<",\"qposadr\":"<<m->jnt_qposadr[i]<<",\"dofadr\":"<<m->jnt_dofadr[i]<<",\"range\":";
    array(trace,m->jnt_range+2*i,2);trace<<'}';
  }
  trace<<"],\"geoms\":[";
  for(int i=0;i<m->ngeom;++i) {
    if(i)trace<<',';trace<<"{\"id\":"<<i<<",\"name\":";quoted(trace,mj_id2name(m,mjOBJ_GEOM,i));
    trace<<",\"body\":";quoted(trace,mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[i]));trace<<'}';
  }
  trace<<"]}\n";trace.flush();initialized=true;
}
}

extern "C" void mj_step(const mjModel* m, mjData* d) {
  using Step=void(*)(const mjModel*,mjData*);
  static auto real=reinterpret_cast<Step>(dlsym(RTLD_NEXT,"mj_step"));
  if(!real)std::abort();
  // SIGINT is an asynchronous exit request. Never integrate another tick while
  // the original GUI/process owner is still unwinding after a safety failure.
  if(safety_stopped) {raise(SIGINT);return;}
  if(!initialized)initialize(m);
  if(trace.is_open() && count>=cap) {
    trace<<"{\"kind\":\"stop\",\"reason\":\"PHYSICS_RESERVATION_CAP\",\"steps\":"<<count<<"}\n";trace.flush();raise(SIGINT);return;
  }
  real(m,d);
  if(!trace.is_open())return;
  if(m!=observed_model)std::abort();
  ++count;
  bool stop=false, warning=false, positive=false;
  for(int i=0;i<m->nq;++i)stop|=!std::isfinite(d->qpos[i]);
  for(int i=0;i<m->nv;++i)stop|=!std::isfinite(d->qvel[i]);
  for(int i=0;i<m->nu;++i)stop|=!std::isfinite(d->ctrl[i]);
  bool illegal_contact=false;
  for(int i=0;i<d->ncon;++i) {
    const char* a=mj_id2name(m,mjOBJ_GEOM,d->contact[i].geom1);
    const char* b=mj_id2name(m,mjOBJ_GEOM,d->contact[i].geom2);
    auto foot=[](const char* s) { return s && (std::string(s)=="FL" || std::string(s)=="FR" || std::string(s)=="RL" || std::string(s)=="RR"); };
    bool allowed=(a && std::string(a)=="floor" && foot(b)) || (b && std::string(b)=="floor" && foot(a));
    illegal_contact|=!allowed;
  }
  const char* safety_marker=std::getenv("PI0288_CONTACT_SAFETY_MARKER");
  bool contact_stop=illegal_contact && safety_marker && access(safety_marker,F_OK)==0;
  for(int j=0;j<m->njnt;++j) {
    if(!m->jnt_limited[j] || (m->jnt_type[j]!=mjJNT_HINGE && m->jnt_type[j]!=mjJNT_SLIDE))continue;
    double q=d->qpos[m->jnt_qposadr[j]];
    double v=std::max(0.0,std::max(m->jnt_range[2*j]-q,q-m->jnt_range[2*j+1]));
    double eps=m->jnt_type[j]==mjJNT_HINGE?1e-4:1e-6;
    double immediate=m->jnt_type[j]==mjJNT_HINGE?1e-3:1e-4;
    violation_duration[j]=v>eps?violation_duration[j]+m->opt.timestep:0;
    positive|=v>0;warning|=v>eps;stop|=v>immediate || violation_duration[j]>=.020-1e-12;
  }
  // Record every positive violation; ordinary snapshots at native control rate.
  if(d->time+1e-12>=next_sample || positive || stop || illegal_contact) {
    mj_copyData(snapshot,m,d);mj_forward(m,snapshot);
    int tcp=mj_name2id(m,mjOBJ_SITE,"x5_ee");
    if(tcp<0)std::abort();
    trace<<"{\"kind\":\"sample\",\"step\":"<<count<<",\"sim_time_s\":"<<d->time
         <<",\"wall_s\":"<<clock_seconds(true)<<",\"monotonic_s\":"<<clock_seconds(false)<<",\"qpos\":";
    array(trace,d->qpos,m->nq);trace<<",\"qvel\":";array(trace,d->qvel,m->nv);
    trace<<",\"ctrl\":";array(trace,d->ctrl,m->nu);
    trace<<",\"tcp_xyz\":";array(trace,snapshot->site_xpos+3*tcp,3);
    trace<<",\"tcp_R\":";array(trace,snapshot->site_xmat+9*tcp,9);
    trace<<",\"positive_bound\":"<<(positive?"true":"false")<<",\"numerical_warning\":"<<(warning?"true":"false")
         <<",\"joint_stop\":"<<(stop?"true":"false")<<",\"illegal_contact\":"<<(illegal_contact?"true":"false")<<",\"contact_stop\":"<<(contact_stop?"true":"false")<<",\"contact_time_s\":"<<d->time-m->opt.timestep<<",\"contacts\":[";
    for(int i=0;i<d->ncon;++i) {
      if(i)trace<<',';mjtNum force[6];mj_contactForce(m,d,i,force);
      trace<<"{\"geom1\":"<<d->contact[i].geom1<<",\"geom2\":"<<d->contact[i].geom2<<",\"dist\":"<<d->contact[i].dist<<",\"force\":";
      array(trace,force,6);trace<<'}';
    }
    trace<<"],\"native_warnings\":[";
    for(int i=0;i<mjNWARNING;++i) {if(i)trace<<',';trace<<d->warning[i].number;}
    trace<<"]}\n";trace.flush();
    if(d->time+1e-12>=next_sample)next_sample=d->time+.005;
  }
  if(stop || contact_stop) {safety_stopped=true;trace<<"{\"kind\":\"stop\",\"reason\":\""<<(stop?"REGISTERED_JOINT_SAFETY":"ILLEGAL_CONTACT")<<"\",\"steps\":"<<count<<"}\n";trace.flush();raise(SIGINT);}
}
