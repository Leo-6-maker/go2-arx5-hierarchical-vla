// Offline reconstruction from preserved q/qdot/ctrl; no mj_step or model mutation.
#include <mujoco/mujoco.h>
#include <iostream>
#include <iomanip>
void array(const mjtNum* p,int n){std::cout<<'[';for(int i=0;i<n;++i){if(i)std::cout<<',';std::cout<<p[i];}std::cout<<']';}
const char* name(const mjModel* m,int type,int id){auto*n=mj_id2name(m,type,id);return n?n:"UNNAMED";}
int main(int argc,char**argv){
  if(argc!=2)return 2;auto*m=mj_loadModel(argv[1],nullptr);if(!m)return 2;
  auto*d=mj_makeData(m);std::cout<<std::setprecision(17);
  std::cout<<"{\"kind\":\"model\",\"physics_integrations\":0,\"joints\":[";
  for(int j=0;j<m->njnt;++j){if(j)std::cout<<',';
    std::cout<<"{\"name\":\""<<name(m,mjOBJ_JOINT,j)<<"\",\"margin\":"<<m->jnt_margin[j]<<",\"solref\":";array(m->jnt_solref+2*j,2);
    std::cout<<",\"solimp\":";array(m->jnt_solimp+5*j,5);std::cout<<",\"damping\":"<<m->dof_damping[m->jnt_dofadr[j]]<<'}';
  }
  std::cout<<"],\"actuators\":[";
  for(int a=0;a<m->nu;++a){if(a)std::cout<<',';std::cout<<"{\"name\":\""<<name(m,mjOBJ_ACTUATOR,a)<<"\",\"joint_id\":"<<m->actuator_trnid[2*a]<<",\"gain_type\":"<<m->actuator_gaintype[a]<<",\"bias_type\":"<<m->actuator_biastype[a]<<",\"gear\":";array(m->actuator_gear+6*a,6);std::cout<<'}';}
  std::cout<<"]}\n";
  double t;while(std::cin>>t){mj_resetData(m,d);d->time=t;
    for(int i=0;i<m->nq;++i)if(!(std::cin>>d->qpos[i]))return 2;
    for(int i=0;i<m->nv;++i)if(!(std::cin>>d->qvel[i]))return 2;
    for(int i=0;i<m->nu;++i)if(!(std::cin>>d->ctrl[i]))return 2;
    mj_forward(m,d);
    std::cout<<"{\"state_time_s\":"<<t<<",\"qfrc_actuator\":";array(d->qfrc_actuator,m->nv);
    std::cout<<",\"qfrc_passive\":";array(d->qfrc_passive,m->nv);
    std::cout<<",\"qfrc_constraint\":";array(d->qfrc_constraint,m->nv);
    std::cout<<",\"qfrc_bias\":";array(d->qfrc_bias,m->nv);
    std::cout<<",\"qacc\":";array(d->qacc,m->nv);std::cout<<",\"ncon\":"<<d->ncon<<"}\n";
  }
  mj_deleteData(d);mj_deleteModel(m);
}
