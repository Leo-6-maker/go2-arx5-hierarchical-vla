// Finite companion of a CURRENT admitted target owner, not a task runner.
// Link the pinned installed SDK; never load a frozen PIE or share C++ objects
// with the running controller. Only original 12D policy-knot numbers cross IO.
#include "target_sdk_constraints.hpp"
#include <ocs2_mobile_manipulator/MobileManipulatorInterface.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
// recompileLibraries=false can still enter loadModelsIfAvailable. No SDK
// fallback may execute a compiler/subprocess binary. The owner additionally
// supplies a sealed read-only PRIVATE cache copy; frozen vendor files are not
// made writable. This filter applies only to this new companion process.
void no_exec_after_start() {
  const sock_filter code[] = {
    BPF_STMT(BPF_LD+BPF_W+BPF_ABS, offsetof(seccomp_data, arch)),
    BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_KILL_PROCESS),
    BPF_STMT(BPF_LD+BPF_W+BPF_ABS, offsetof(seccomp_data, nr)),
    BPF_JUMP(BPF_JMP+BPF_JGE+BPF_K, 0x40000000U, 0, 1),
    BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_ERRNO | EACCES),
    BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, __NR_execve, 0, 1),
    BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_ERRNO | EACCES),
    BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, __NR_execveat, 0, 1),
    BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_ERRNO | EACCES),
    BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(sizeof(code)/sizeof(code[0])),
                     const_cast<sock_filter*>(code)};
  need(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)==0, "SDK_NO_NEW_PRIVS_FAILED");
  need(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program)==0, "SDK_NO_EXEC_FILTER_FAILED");
}
bool input(void* data, size_t count, bool boundary=false) {
  auto* p=static_cast<char*>(data);size_t done=0;
  while(done<count) {
    const auto n=::read(STDIN_FILENO,p+done,count-done);
    if(n<0 && errno==EINTR)continue;
    if(n==0 && done==0 && boundary)return false;
    need(n>0,"SDK_TRUNCATED_REQUEST");done+=static_cast<size_t>(n);
  }
  return true;
}
template<class T> void append(std::vector<char>& out,const T& value) {
  const auto* p=reinterpret_cast<const char*>(&value);out.insert(out.end(),p,p+sizeof(value));
}
void output(int fd,const std::vector<char>& bytes) {
  size_t done=0;
  while(done<bytes.size()) {
    auto n=::write(fd,bytes.data()+done,bytes.size()-done);
    if(n<0 && errno==EINTR)continue;
    need(n>0,"SDK_RESPONSE_PIPE_FAILED");done+=static_cast<size_t>(n);
  }
}
std::vector<char> magic(const char* value) {return std::vector<char>(value,value+8);}
}  // namespace

int main(int argc,char** argv) {
  try {
    // argv is prepared by the existing admitted owner, never shell-evaluated.
    // task/cache/URDF/base/left-tip/right-tip/response-fd/request-cap/mode
    need(argc==10,"SDK_EXPLICIT_PINNED_ARGUMENTS_REQUIRED");
    const uint32_t endian=1;
    need(*reinterpret_cast<const char*>(&endian)==1 && sizeof(double)==8 &&
         std::numeric_limits<double>::is_iec559,"SDK_ORIGINAL_X86_64_F64_WIRE");
    size_t consumed=0;int fd=std::stoi(argv[7],&consumed);
    need(consumed==std::strlen(argv[7]) && fd>2 && fd<1024,"SDK_DEDICATED_INHERITED_RESPONSE_FD");
    const int maximum=std::stoi(argv[8],&consumed);
    need(consumed==std::strlen(argv[8]) && maximum>0 && maximum<=80,"SDK_ORIGINAL_FINITE_REQUEST_CAP");
    ocs2::mobile_manipulator::FrameOverrides frames;
    auto frame=[](const char* value){return std::string(value)=="-" ? std::string{} : std::string(value);};
    frames.baseFrame=frame(argv[4]);frames.eeFrame=frame(argv[5]);frames.eeFrame1=frame(argv[6]);
    const bool initialize=std::string(argv[9])=="ORIGINAL_SDK_INITIALIZE";
    need(initialize || std::string(argv[9])=="SEALED_CACHE_LOAD","SDK_EXPLICIT_INITIALIZATION_MODE");
    // Cold qualification may explicitly realize the ORIGINAL SDK's AD model
    // using its unchanged original task file. This is not a controller build.
    // The owner reserves compiler CPU/IO and a fresh private output directory.
    // No implicit fallback is permitted in a task's sealed-cache load mode.
    if(!initialize)no_exec_after_start();
    ocs2::mobile_manipulator::MobileManipulatorInterface sdk(argv[1],argv[2],argv[3],frames);
    if(initialize)no_exec_after_start();  // all later requests forbid exec too
    const auto& info=sdk.getManipulatorModelInfo();
    need(info.stateDim==12 && info.inputDim==12 &&
         static_cast<int>(info.manipulatorModelType)==3,"SDK_FROZEN_MODEL_TYPE_AND_DIMENSIONS");
    dec1::OriginalSdkConstraints reader(sdk.getOptimalControlProblem());
    auto ready=magic("D1SDKI01");
    for(uint32_t v:{12,12,3})append(ready,v);
    output(fd,ready);  // SDK stdout/stderr NEVER carry machine replies
    for(int request=0;request<maximum;++request) {
      char tag[8];if(!input(tag,sizeof(tag),true))return 0;
      need(std::memcmp(tag,"D1SDKC01",8)==0,"SDK_UNKNOWN_REQUEST_WIRE");
      uint32_t count;input(&count,sizeof(count));need(count>=2 && count<=4096,"SDK_BOUNDED_KNOT_COUNT");
      std::vector<dec1::SdkKnot> knots;knots.reserve(count);
      for(uint32_t i=0;i<count;++i) {
        dec1::SdkKnot k;input(&k.time,sizeof(k.time));
        k.state=ocs2::vector_t(12);k.input=ocs2::vector_t(12);
        for(int j=0;j<12;++j)input(&k.state[j],sizeof(double));
        for(int j=0;j<12;++j)input(&k.input[j],sizeof(double));
        knots.push_back(std::move(k));
      }
      const auto samples=reader.sample(knots);
      auto bytes=magic("D1SDKR01");append(bytes,count);
      for(const auto& s:samples) {
        need(s.self_collision.size()<=4096,"SDK_BOUNDED_CONSTRAINT_VECTOR");
        append(bytes,s.time);append(bytes,s.joint_limit_cost);
        append(bytes,static_cast<uint32_t>(s.self_collision.size()));
        for(decltype(s.self_collision.size()) j=0;j<s.self_collision.size();++j)append(bytes,s.self_collision[j]);
        need(bytes.size()<=4*1024*1024,"SDK_BOUNDED_WHOLE_RESPONSE");
      }
      output(fd,bytes);  // one complete validated batch, no partial verdict
    }
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"DEC1_ORIGINAL_SDK_CLIENT_FAILED:"<<error.what()<<'\n';return 1;
  }
}
