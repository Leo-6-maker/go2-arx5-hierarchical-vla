// Read-only, zero-integration qualification of the delivered home keyframe.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "native_home.hpp"

static const char* name(const mjModel* m, int type, int id) {
  auto* n=mj_id2name(m,type,id); return n?n:"UNNAMED";
}
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  char error[2048]="MJB load failed";
  std::string path=argv[1];auto* m=path.size()>4 && path.substr(path.size()-4)==".mjb"?mj_loadModel(argv[1],nullptr):mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 2;}
  int key=mj_name2id(m,mjOBJ_KEY,"home");
  if(key<0){std::cerr<<"Missing home";return 2;}
  auto* d=mj_makeData(m); PI0288ResetHome(m,d);
  int size=mj_stateSize(m,mjSTATE_INTEGRATION);
  std::vector<mjtNum> before(size),after(size);
  mj_getState(m,d,before.data(),mjSTATE_INTEGRATION);
  d->qpos[0]=42;d->qvel[0]=7;d->ctrl[0]=3;d->time=9;
  PI0288ResetHome(m,d);mj_getState(m,d,after.data(),mjSTATE_INTEGRATION);
  bool reset_exact=std::memcmp(before.data(),after.data(),size*sizeof(mjtNum))==0;
  int violations=0,illegal=0; bool finite=true;
  for(int i=0;i<m->nq;++i)finite &= std::isfinite(d->qpos[i]);
  for(int i=0;i<m->nv;++i)finite &= std::isfinite(d->qvel[i]) && std::isfinite(d->qacc[i]);
  std::cout<<std::setprecision(17)<<"{\"physics_integrations\":0,\"reset_exact\":"<<(reset_exact?"true":"false")<<",\"mujoco_version\":\""<<mj_versionString()<<"\",\"time\":"<<d->time<<",\"key\":\"home\",\"base_xyz\":["<<d->qpos[0]<<','<<d->qpos[1]<<','<<d->qpos[2]<<"],\"joints\":[";
  bool first=true;
  for(int j=0;j<m->njnt;++j) {
    if(!m->jnt_limited[j] || (m->jnt_type[j]!=mjJNT_HINGE && m->jnt_type[j]!=mjJNT_SLIDE))continue;
    if(!first)std::cout<<',';first=false;
    double q=d->qpos[m->jnt_qposadr[j]],lo=m->jnt_range[2*j],hi=m->jnt_range[2*j+1];
    double v=std::max(0.0,std::max(lo-q,q-hi));violations+=v>0;
    int a=m->jnt_dofadr[j];
    std::cout<<"{\"name\":\""<<name(m,mjOBJ_JOINT,j)<<"\",\"q\":"<<q<<",\"lower\":"<<lo<<",\"upper\":"<<hi<<",\"lower_margin\":"<<q-lo<<",\"upper_margin\":"<<hi-q<<",\"qvel\":"<<d->qvel[a]<<",\"qacc_at_zero_ctrl\":"<<d->qacc[a]<<'}';
  }
  std::cout<<"],\"feet\":[";first=true;
  int floor=mj_name2id(m,mjOBJ_GEOM,"floor");if(floor<0)return 2;
  for(auto* foot:{"FL","FR","RL","RR"}) {
    int g=mj_name2id(m,mjOBJ_GEOM,foot);if(g<0)return 2;
    if(!first)std::cout<<',';first=false;
    mjtNum segment[6]; double distance=mj_geomDistance(m,d,g,floor,1,segment);
    std::cout<<"{\"name\":\""<<foot<<"\",\"center_z\":"<<d->geom_xpos[3*g+2]<<",\"floor_distance\":"<<distance<<'}';
  }
  std::cout<<"],\"contacts\":[";
  for(int i=0;i<d->ncon;++i) {
    if(i)std::cout<<',';auto& c=d->contact[i];
    std::string a=name(m,mjOBJ_GEOM,c.geom[0]),b=name(m,mjOBJ_GEOM,c.geom[1]);
    auto foot=[](const std::string& s){return s=="FL"||s=="FR"||s=="RL"||s=="RR";};
    bool allowed=(a=="floor"&&foot(b))||(b=="floor"&&foot(a));
    illegal+=!allowed && c.dist<=0;
    mjtNum force[6];mj_contactForce(m,d,i,force);
    std::cout<<"{\"geom_a\":\""<<a<<"\",\"geom_b\":\""<<b<<"\",\"body_a\":\""<<name(m,mjOBJ_BODY,m->geom_bodyid[c.geom[0]])<<"\",\"body_b\":\""<<name(m,mjOBJ_BODY,m->geom_bodyid[c.geom[1]])<<"\",\"distance\":"<<c.dist<<",\"allowed\":"<<(allowed?"true":"false")<<",\"normal_force\":"<<force[0]<<'}';
  }
  std::cout<<"],\"ctrl\":[";
  for(int i=0;i<m->nu;++i){if(i)std::cout<<',';std::cout<<d->ctrl[i];}
  std::cout<<"],\"invalid_joint_count\":"<<violations<<",\"illegal_touching_contacts\":"<<illegal<<",\"finite\":"<<(finite?"true":"false")<<",\"warnings\":[";
  for(int i=0;i<mjNWARNING;++i){if(i)std::cout<<',';std::cout<<d->warning[i].number;}
  std::cout<<"]}\n";
  mj_deleteData(d);mj_deleteModel(m);
  return violations||illegal||!finite||!reset_exact?1:0;
}
