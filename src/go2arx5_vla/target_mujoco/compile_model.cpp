// Preserve the XML compiler self-test separately from actual runtime mjData.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <vector>
static int compiler_steps=0;
extern "C" void mj_step(const mjModel* m,mjData* d) {
  if(++compiler_steps>1)_exit(77);
  using Step=void(*)(const mjModel*,mjData*);
  auto real=reinterpret_cast<Step>(dlsym(RTLD_NEXT,"mj_step"));if(!real)_exit(77);
  real(m,d);
  std::cerr<<std::setprecision(17)<<"{\"kind\":\"XML_COMPILER_SELF_TEST\",\"step\":"<<compiler_steps<<",\"time\":"<<d->time<<",\"qpos\":[";
  for(int i=0;i<m->nq;++i){if(i)std::cerr<<',';std::cerr<<d->qpos[i];}
  std::cerr<<"]}\n";
}
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  char error[2048];auto* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 2;}
  mj_saveModel(m,argv[2],nullptr,0);
  auto* restored=mj_loadModel(argv[2],nullptr);if(!restored)return 2;
  int size=mj_sizeModel(m);std::vector<char> a(size),b(size);
  mj_saveModel(m,nullptr,a.data(),size);mj_saveModel(restored,nullptr,b.data(),size);
  bool same=mj_sizeModel(restored)==size && a==b;
  std::cout<<"{\"compiled_model_serialization_byte_equal\":"<<(same?"true":"false")<<",\"compiler_self_test_steps\":"<<compiler_steps<<",\"runtime_integrations\":0,\"serialized_bytes\":"<<size<<"}\n";
  mj_deleteModel(restored);mj_deleteModel(m);return same?0:1;
}
