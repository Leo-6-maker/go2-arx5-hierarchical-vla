// Additional boundary observer; frozen native control, physics and old guard stay.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <signal.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <unistd.h>
#include <algorithm>

namespace {
bool ready=false,stopped=false,armed=false;
double zlo,zhi,plo,phi,rlo,rhi,radius,flight_max,unsupported=0;
double origin_x=0,origin_y=0;
std::vector<int> actuators;std::vector<double> efforts;
int mount=-1,floor_geom=-1;int feet[4];mjData* snapshot=nullptr;
const char* stop_path=nullptr;const char* armed_path=nullptr;
std::ofstream trace;
void stop(const char* reason,const mjData* d) {
  stopped=true;
  if(stop_path) {std::ofstream out(stop_path);out<<std::setprecision(17)
    <<"{\"reason\":\""<<reason<<"\",\"time_s\":"<<d->time<<",\"native_dt_s\":0.002}";out.flush();}
  raise(SIGINT);
}
void setup(const mjModel* m) {
  const char* file=std::getenv("MOBILE_SAFETY_BINDING");
  armed_path=std::getenv("MOBILE_SAFETY_ARMED");stop_path=std::getenv("MOBILE_SAFETY_STOP");
  const char* log=std::getenv("MOBILE_SAFETY_TRACE");
  if(!file||!armed_path||!stop_path||!log||std::abs(m->opt.timestep-.002)>1e-12)std::abort();
  std::ifstream input(file);std::string version;input>>version;
  if(version!="MOBILE_SAFETY_V1")std::abort();
  input>>zlo>>zhi>>plo>>phi>>rlo>>rhi>>radius>>flight_max;
  int n;input>>n;if(n!=18||radius!=1.||!(zlo<zhi&&plo<phi&&rlo<rhi&&flight_max>0))std::abort();
  for(int k=0;k<n;++k) {
    std::string name;double limit;input>>name>>limit;
    int joint=mj_name2id(m,mjOBJ_JOINT,name.c_str()),found=-1;
    for(int a=0;a<m->nu;++a)if(m->actuator_trnid[2*a]==joint&&m->actuator_trntype[a]==mjTRN_JOINT)found=a;
    if(!input||joint<0||found<0||limit<=0||m->actuator_gear[6*found]!=1.)std::abort();
    actuators.push_back(found);efforts.push_back(limit);
  }
  mount=mj_name2id(m,mjOBJ_BODY,"x5_base_link");floor_geom=mj_name2id(m,mjOBJ_GEOM,"floor");
  const char* names[]={"FL","FR","RL","RR"};
  for(int k=0;k<4;++k) {feet[k]=mj_name2id(m,mjOBJ_GEOM,names[k]);if(feet[k]<0)std::abort();}
  if(mount<0||floor_geom<0)std::abort();
  snapshot=mj_makeData(m);if(!snapshot)std::abort();
  trace.open(log,std::ios::app);if(!trace)std::abort();trace<<std::setprecision(17);
  trace<<"{\"kind\":\"mobile_guard_loaded\",\"physics_dt_s\":0.002,\"torque_joints\":18,\"frozen_controller_changed\":false}\n";trace.flush();
  ready=true;
}
int evaluate(const mjModel* m,const mjData* d) {
  if(!armed)return 0;
  for(int k=0;k<m->nq;++k)if(!std::isfinite(d->qpos[k]))return 1;
  for(int k=0;k<m->nv;++k)if(!std::isfinite(d->qvel[k]))return 1;
  mj_copyData(snapshot,m,d);mj_forward(m,snapshot);
  const double* p=snapshot->xpos+3*mount;const double* R=snapshot->xmat+9*mount;
  const double pitch=std::asin(std::max(-1.,std::min(1.,-R[6]))),roll=std::atan2(R[7],R[8]);
  if(!std::isfinite(p[0]+p[1]+p[2]+pitch+roll))return 1;
  if(p[2]<zlo||p[2]>zhi)return 2;
  if(pitch<plo||pitch>phi||roll<rlo||roll>rhi)return 3;
  if(std::hypot(p[0]-origin_x,p[1]-origin_y)>radius)return 4;
  for(size_t k=0;k<actuators.size();++k) {
    const int a=actuators[k];
    if(!std::isfinite(d->ctrl[a])||!std::isfinite(d->actuator_force[a]))return 5;
    if(std::abs(d->ctrl[a])>efforts[k]||std::abs(d->actuator_force[a])>efforts[k])return 6;
  }
  int contacts=0;bool touching[4]={false,false,false,false};
  for(int c=0;c<snapshot->ncon;++c) {
    const mjContact& contact=snapshot->contact[c];
    if(contact.geom[0]!=floor_geom&&contact.geom[1]!=floor_geom)continue;
    const int other=contact.geom[0]==floor_geom?contact.geom[1]:contact.geom[0];bool foot=false;
    for(int k=0;k<4;++k)if(other==feet[k]) {foot=true;touching[k]=true;}
    if(!foot)return 7; // Normal leg swing is allowed; a non-foot floor collision is not.
    ++contacts;
  }
  unsupported=contacts?0.:unsupported+m->opt.timestep;
  if(unsupported>flight_max)return 8;
  const auto tick=std::llround(d->time/.002);
  if(tick%25==0) {
    trace<<"{\"kind\":\"mobile_actual\",\"tick\":"<<tick<<",\"time_s\":"<<d->time
      <<",\"mount_xyz\":["<<p[0]<<','<<p[1]<<','<<p[2]<<"],\"pitch\":"<<pitch
      <<",\"roll\":"<<roll<<",\"foot_contacts\":"<<contacts<<",\"unsupported_s\":"<<unsupported<<",\"stance_tangent_speed_m_s\":[";
    for(int k=0;k<4;++k) {mjtNum velocity[6];mj_objectVelocity(m,snapshot,mjOBJ_GEOM,feet[k],velocity,0);
      if(k)trace<<',';if(touching[k])trace<<std::hypot(velocity[3],velocity[4]);else trace<<"null";}
    trace<<"]}\n";trace.flush();
  }
  return 0;
}
}

// Fault injection uses a private model/data copy without calling mj_step.
extern "C" int mobile_safety_evaluate_v1(const mjModel* m,const mjData* d) {
  if(!ready)setup(m);
  if(!armed) {mj_copyData(snapshot,m,d);mj_forward(m,snapshot);origin_x=snapshot->xpos[3*mount];origin_y=snapshot->xpos[3*mount+1];armed=true;}
  return evaluate(m,d);
}

extern "C" void mj_step(const mjModel* m,mjData* d) {
  using Step=void(*)(const mjModel*,mjData*);static auto next=reinterpret_cast<Step>(dlsym(RTLD_NEXT,"mj_step"));
  if(!next)std::abort();if(stopped) {raise(SIGINT);return;}
  if(!ready)setup(m);
  if(!armed&&access(armed_path,F_OK)==0) {
    std::ifstream in(armed_path);std::string version;in>>version>>origin_x>>origin_y;
    if(version!="MOBILE_ARMED_V1"||!in)std::abort();armed=true;
  }
  next(m,d);
  const int failure=evaluate(m,d);
  const char* reasons[]={"OK","NONFINITE_ACTUAL","MOUNT_HEIGHT","BODY_ATTITUDE","WORKSPACE_EXIT",
    "NONFINITE_ACTUATOR","NATIVE_ACTUATOR_EFFORT","NON_FOOT_FLOOR_CONTACT","GAIT_SUPPORT_LOST"};
  if(failure)stop(reasons[failure],d);
}
