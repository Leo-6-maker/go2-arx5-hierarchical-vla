// UNINSTALLED candidate. The historical zero-false-block screen FAILED.
// Separate observer only: no controller, reference, actuator or model writes.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

struct CloseFenceResultV1 {
  double clearance_m[2];
  double actual_minus_receiver_z_m;
  double conditional_upper_error_m;
  double receiver_age_s;
  int overlaps_xy[2];
  int alarm;
};

namespace {
constexpr double dt=.002, clearance=.003, error=-.005, reference_rate=.05;
bool finite_state(const mjModel* m,const mjData* d) {
  for(int k=0;k<m->nq;++k)if(!std::isfinite(d->qpos[k]))return false;
  for(int k=0;k<m->nv;++k)if(!std::isfinite(d->qvel[k]))return false;
  for(int k=0;k<m->nu;++k)if(!std::isfinite(d->ctrl[k]))return false;
  return std::isfinite(d->time);
}
int evaluate(const mjModel* m,const mjData* d,double reference_s,double reference_z,CloseFenceResultV1* out) {
  if(!m||!d||!out||mjcb_control||mjcb_sensor||mjcb_passive||mjcb_act_dyn||mjcb_act_gain||mjcb_act_bias)return 2;
  std::memset(out,0,sizeof(*out));
  const double age=d->time-reference_s;
  if(!finite_state(m,d)||std::abs(m->opt.timestep-dt)>1e-12||!std::isfinite(reference_s)
      ||!std::isfinite(reference_z)||age< -1e-8||age>.05+1e-8)return 2;
  const int table=mj_name2id(m,mjOBJ_GEOM,"task_table"),tcp=mj_name2id(m,mjOBJ_SITE,"x5_ee");
  const int pads[]={mj_name2id(m,mjOBJ_GEOM,"task_left_pad"),mj_name2id(m,mjOBJ_GEOM,"task_right_pad")};
  if(table<0||tcp<0||pads[0]<0||pads[1]<0||m->geom_type[table]!=mjGEOM_BOX)return 2;
  for(int g:pads)if(m->geom_type[g]!=mjGEOM_BOX)return 2;
  std::unique_ptr<mjData,decltype(&mj_deleteData)> copy(mj_makeData(m),mj_deleteData);
  if(!copy)return 2;
  mj_copyData(copy.get(),m,d);mj_forward(m,copy.get()); // Passive FK only.
  const double* table_p=copy->geom_xpos+3*table;const double* table_R=copy->geom_xmat+9*table;
  out->receiver_age_s=std::max(0.,age);
  out->actual_minus_receiver_z_m=copy->site_xpos[3*tcp+2]-reference_z;
  out->conditional_upper_error_m=out->actual_minus_receiver_z_m+reference_rate*out->receiver_age_s;
  if(!std::isfinite(out->conditional_upper_error_m))return 2;
  for(int k=0;k<2;++k) {
    const int g=pads[k];double lo[]={INFINITY,INFINITY,INFINITY},hi[]={-INFINITY,-INFINITY,-INFINITY};
    const double* p=copy->geom_xpos+3*g;const double* R=copy->geom_xmat+9*g;
    for(int a=-1;a<=1;a+=2)for(int b=-1;b<=1;b+=2)for(int c=-1;c<=1;c+=2) {
      const double local[]={a*m->geom_size[3*g],b*m->geom_size[3*g+1],c*m->geom_size[3*g+2]};
      double relative[3];
      for(int i=0;i<3;++i) {relative[i]=p[i]-table_p[i];for(int j=0;j<3;++j)relative[i]+=R[3*i+j]*local[j];}
      for(int i=0;i<3;++i) {double value=0;for(int j=0;j<3;++j)value+=table_R[3*j+i]*relative[j];
        if(!std::isfinite(value))return 2;lo[i]=std::min(lo[i],value);hi[i]=std::max(hi[i],value);}
    }
    out->clearance_m[k]=lo[2]-m->geom_size[3*table+2];
    out->overlaps_xy[k]=hi[0]>=-m->geom_size[3*table]&&hi[1]>=-m->geom_size[3*table+1]
      &&lo[0]<=m->geom_size[3*table]&&lo[1]<=m->geom_size[3*table+1];
    out->alarm|=out->overlaps_xy[k]&&out->clearance_m[k]<=clearance&&out->conditional_upper_error_m<=error;
  }
  return out->alarm?1:0;
}

bool latched=false,configured=false,armed=false;
const char *armed_path,*witness_path,*stop_path;
long long last_witness_tick=-1;
void configure() {
  armed_path=std::getenv("SLA_CLOSE_FENCE_ARMED");witness_path=std::getenv("SLA_CLOSE_FENCE_WITNESS");
  stop_path=std::getenv("SLA_CLOSE_FENCE_STOP");
  if(!armed_path||!witness_path||!stop_path)std::abort();
  configured=true;
}
void latch(const char* reason,const mjData* d,const CloseFenceResultV1& value) {
  latched=true;
  std::ofstream out(stop_path);if(!out)std::abort();out<<std::setprecision(17)
    <<"{\"reason\":\""<<reason<<"\",\"time_s\":"<<d->time
    <<",\"receiver_age_s\":"<<value.receiver_age_s
    <<",\"conditional_upper_error_m\":"<<value.conditional_upper_error_m
    <<",\"guard_contract\":\"CLOSE_CONTACT_RECEIVER_BOUND_CANDIDATE_V1\""
    <<",\"controller_selected_reference_known\":false,\"WBC_changed\":false}";out.flush();
  if(!out)std::abort();raise(SIGINT);
}
}

// Read-only ABI for numerical comparison with historical recorded states.
extern "C" int close_contact_fence_evaluate_v1(const mjModel* m,const mjData* d,
    double reference_s,double reference_z,CloseFenceResultV1* out) {
  try{return evaluate(m,d,reference_s,reference_z,out);}catch(...){return 2;}
}

// Future, separately admitted profile only. This source is never auto-installed.
extern "C" void mj_step(const mjModel* m,mjData* d) {
  using Step=void(*)(const mjModel*,mjData*);
  static auto next=reinterpret_cast<Step>(dlsym(RTLD_NEXT,"mj_step"));
  if(!next)std::abort();
  if(latched){raise(SIGINT);return;} // SIGINT is asynchronous: no later integration.
  if(!configured)configure();
  const bool marker_present=access(armed_path,F_OK)==0;
  armed=armed||marker_present;
  if(armed) {
    CloseFenceResultV1 value{};std::ifstream input(witness_path);
    std::string version,wire,extra;long long tick=-1;double seconds=0,z=0;
    input>>version>>tick>>seconds>>z>>wire;
    bool valid=marker_present&&bool(input)&&version=="SLA_CLOSE_EXACT_RECEIVER_BOUNDARY_V1"&&tick>=last_witness_tick
      &&wire.size()==64&&std::all_of(wire.begin(),wire.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');})
      &&std::abs(seconds-dt*tick)<=1e-8&&!(input>>extra);
    int code=valid?close_contact_fence_evaluate_v1(m,d,seconds,z,&value):2;
    if(code){latch(code==1?"CANDIDATE_PAD_CLEARANCE_TRACKING":"INVALID_OR_STALE_RECEIVER_WITNESS",d,value);return;}
    last_witness_tick=tick;
  }
  next(m,d); // Original guards remain in the chain. No command/reference edits.
}
