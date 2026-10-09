// Zero-integration audit of the delivered model and observation-copy invariance.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  char error[2048];auto* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 2;}
  auto* d=mj_makeData(m);auto* copy=mj_makeData(m);
  int n=mj_stateSize(m,mjSTATE_INTEGRATION);std::vector<mjtNum> before(n),after(n);
  mj_getState(m,d,before.data(),mjSTATE_INTEGRATION);
  mj_copyData(copy,m,d);mj_forward(m,copy);
  mj_getState(m,d,after.data(),mjSTATE_INTEGRATION);
  bool unchanged=std::memcmp(before.data(),after.data(),n*sizeof(mjtNum))==0;
  std::cout<<std::setprecision(17)<<"{\"physics_integrations\":0,\"mujoco_version\":\""<<mj_versionString()<<"\",\"timestep\":"<<m->opt.timestep<<",\"nq\":"<<m->nq<<",\"nv\":"<<m->nv<<",\"nu\":"<<m->nu<<",\"private_snapshot_original_integration_state_unchanged\":"<<(unchanged?"true":"false")<<",\"initial_joint_bounds\":[";
  bool first=true;int violations=0;
  for(int j=0;j<m->njnt;++j) {
    if(!m->jnt_limited[j] || (m->jnt_type[j]!=mjJNT_HINGE && m->jnt_type[j]!=mjJNT_SLIDE))continue;
    if(!first)std::cout<<',';first=false;
    double q=d->qpos[m->jnt_qposadr[j]],lo=m->jnt_range[2*j],hi=m->jnt_range[2*j+1];
    double v=std::max(0.0,std::max(lo-q,q-hi));violations+=v>0;
    std::cout<<"{\"joint\":\""<<mj_id2name(m,mjOBJ_JOINT,j)<<"\",\"q\":"<<q<<",\"lower\":"<<lo<<",\"upper\":"<<hi<<",\"positive_violation\":"<<v<<'}';
  }
  std::cout<<"],\"invalid_default_joint_count\":"<<violations<<",\"keyframes\":[";
  for(int k=0;k<m->nkey;++k) {
    if(k)std::cout<<',';
    std::cout<<"{\"name\":\""<<mj_id2name(m,mjOBJ_KEY,k)<<"\",\"qpos\":[";
    for(int i=0;i<m->nq;++i){if(i)std::cout<<',';std::cout<<m->key_qpos[k*m->nq+i];}
    std::cout<<"]}";
  }
  std::cout<<"]}\n";
  mj_deleteData(copy);mj_deleteData(d);mj_deleteModel(m);
  return unchanged?0:1;
}
