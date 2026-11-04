// Independent finite gripper + actual-state recorder. No original controller,
// arm/leg actuator, qpos, qvel, constraint, external force or callback edits.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <signal.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <unistd.h>

namespace {
int joint[2], actuator[2];
double target[2];
bool initialized=false,has_command=false;
uint64_t steps=0,sequence=0;
double opening=1.,expiry=-1.;
const char *command_path=nullptr,*state_path=nullptr,*quota_path=nullptr;
template<class T> void put(std::ostream& f,const T& value) {f.write(reinterpret_cast<const char*>(&value),sizeof(T));}
void fail(const char* reason) {
 if(const char* p=std::getenv("GR00T_TASK_ERROR")){std::ofstream f(p,std::ios::app);f<<reason<<'\n';}
 raise(SIGINT);
}
void initialize(const mjModel* m,const mjData* d) {
 command_path=std::getenv("GR00T_TASK_GRIPPER_COMMAND");state_path=std::getenv("GR00T_TASK_STATE");
 quota_path=std::getenv("GR00T_TASK_STEP_QUOTA");
 if(!command_path || !state_path || m->nu!=20 || m->nq!=34 || m->nv!=32)std::abort();
 for(int i=0;i<2;++i) {
  const char* name=i?"task_right_slide":"task_left_slide";
  joint[i]=mj_name2id(m,mjOBJ_JOINT,name);
  actuator[i]=mj_name2id(m,mjOBJ_ACTUATOR,i?"task_right_motor":"task_left_motor");
  if(joint[i]<0 || actuator[i]!=18+i || m->jnt_type[joint[i]]!=mjJNT_SLIDE || m->actuator_gear[6*actuator[i]]!=1)std::abort();
  target[i]=d->qpos[m->jnt_qposadr[joint[i]]];
 }
 initialized=true;
}
uint64_t quota() {
 uint64_t value=0;std::ifstream f(quota_path?quota_path:"");f>>value;
 if(!f || value==0 || value>20000) {fail("INVALID_ACTUAL_STEP_QUOTA");return 0;}
 return value;
}
void read_command(const mjData* d) {
 std::ifstream f(command_path,std::ios::binary);if(!f)return;
 char magic[8];uint64_t seq;double source,value,until;
 f.read(magic,8);f.read(reinterpret_cast<char*>(&seq),8);f.read(reinterpret_cast<char*>(&source),8);
 f.read(reinterpret_cast<char*>(&value),8);f.read(reinterpret_cast<char*>(&until),8);
 if(!f || f.peek()!=EOF || std::memcmp(magic,"GRTGRIP1",8) || !std::isfinite(source) || !std::isfinite(value)
    || !std::isfinite(until) || value<.002/.044 || value>.042/.044 || until<source || until-source>.400000001 || source>d->time+.025) {
  fail("INVALID_GRIPPER_PACKET");return;
 }
 if(has_command && seq==sequence)return;
 if((!has_command && seq!=0) || (has_command && seq!=sequence+1) || d->time-source>.025 || until<d->time) {
  fail("STALE_OR_REORDERED_GRIPPER_PACKET");return;
 }
 has_command=true;sequence=seq;opening=value;expiry=until;
}
void record(const mjModel* m,const mjData* d) {
 std::string temporary=std::string(state_path)+".pending";
 std::ofstream f(temporary,std::ios::binary|std::ios::trunc);if(!f)std::abort();
 f.write("GRTSTATE",8);put(f,steps);put(f,d->time);
 uint32_t nq=m->nq,nv=m->nv,nu=m->nu,ncon=d->ncon;put(f,nq);put(f,nv);put(f,nu);
 f.write(reinterpret_cast<const char*>(d->qpos),8*nq);f.write(reinterpret_cast<const char*>(d->qvel),8*nv);f.write(reinterpret_cast<const char*>(d->ctrl),8*nu);
 put(f,ncon);
 for(int i=0;i<d->ncon;++i) {
  const auto& c=d->contact[i];uint32_t a=c.geom1,b=c.geom2;put(f,a);put(f,b);put(f,c.dist);
  mjtNum force[6];mj_contactForce(m,d,i,force);
  f.write(reinterpret_cast<const char*>(force),48);f.write(reinterpret_cast<const char*>(c.frame),72);f.write(reinterpret_cast<const char*>(c.pos),24);
 }
 f.close();if(!f)std::abort();
 std::ifstream input(temporary,std::ios::binary);
 std::string payload((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
 static std::ofstream journal(std::string(state_path)+".journal",std::ios::binary|std::ios::app);
 uint32_t length=payload.size();put(journal,length);journal.write(payload.data(),length);journal.flush();
 if(!journal || std::rename(temporary.c_str(),state_path))std::abort();
}
}

extern "C" void mj_step(const mjModel* m,mjData* d) {
 using Step=void(*)(const mjModel*,mjData*);
 static auto real=reinterpret_cast<Step>(dlsym(RTLD_NEXT,"mj_step"));if(!real)std::abort();
 if(!initialized)initialize(m,d);
 const auto limit=quota();
 if(!limit || steps>=limit) {fail("STEP_QUOTA_OVERRUN_REJECT");return;}
 read_command(d);
 for(int i=0;i<2;++i) {
  double q=d->qpos[m->jnt_qposadr[joint[i]]],dq=d->qvel[m->jnt_dofadr[joint[i]]];
  if(!std::isfinite(q) || !std::isfinite(dq) || std::abs(dq)>.25) {fail("GRIPPER_FINITE_SPEED_SAFETY");return;}
  double requested=has_command && d->time<=expiry ? opening*.044 : target[i];
  target[i]+=std::clamp(requested-target[i],-.04*m->opt.timestep,.04*m->opt.timestep);
  d->ctrl[actuator[i]]=std::clamp(400*(target[i]-q)-3*dq,-8.,8.);
 }
 double before=d->time;real(m,d);
 // A latched inner safety observer can reject the call without advancing time.
 // Do not append a duplicate packet or pretend that a new transition occurred.
 if(d->time<=before)return;
 ++steps;record(m,d);
 if(steps==limit) {
  using Pause=void(*)(uint64_t);
  static auto pause=reinterpret_cast<Pause>(dlsym(RTLD_DEFAULT,"GR00TTaskPause"));
  if(!pause) {fail("EXACT_STEP_PAUSE_HOOK_MISSING");return;}
  pause(steps);
 }
}
