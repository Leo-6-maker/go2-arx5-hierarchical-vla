// Dedicated gripper lease, observed by the same receiver that applies the two slide targets.
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace student_grip {
struct Packet {
  char magic[8];
  uint64_t sequence;
  double source, opening, expiry;
};
static_assert(sizeof(Packet) == 40);
struct Envelope { char epoch[64]; Packet command; };
static_assert(sizeof(Envelope) == 104);

struct Session {
  std::string dir, epoch;
  FILE* journal=nullptr;
  uint64_t stop_tick=0;
  bool active=false, stopped=false;
  Packet last{};
  void event(const char* kind, uint64_t tick, const double* target, const double* measured) {
    const int ok=std::fprintf(journal,"%s,%s,%llu,%d,%llu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
      kind,epoch.c_str(),static_cast<unsigned long long>(tick),int(active),
      static_cast<unsigned long long>(last.sequence),last.source,last.opening,last.expiry,
      target[0],target[1],measured[0],measured[1]);
    if(ok<0 || std::fflush(journal)!=0) std::abort();
  }
  void reset(const char* path,const char* token,uint64_t cap,const double* target) {
    if(!path || !token || std::strlen(token)!=64 || cap==0 || journal) std::abort();
    dir=path; epoch=token;stop_tick=cap;
    if(epoch.find_first_not_of("0123456789abcdef")!=std::string::npos) std::abort();
    struct stat st{};
    if(lstat(path,&st)!=0 || !S_ISDIR(st.st_mode) || st.st_uid!=getuid() || (st.st_mode&0077)) std::abort();
    const int fd=open((dir+"/receiver.csv").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0 || !(journal=fdopen(fd,"w")))std::abort();
    event("PHYSICAL_RESET",0,target,target);
  }
  bool receive(uint64_t tick,double time,const double* target,const double* measured){
    if(stopped || tick>=stop_tick)return false;
    int fd=open((dir+"/command.bin").c_str(),O_RDONLY|O_NOFOLLOW|O_NONBLOCK);
    if(fd<0)return errno==ENOENT && !active;
    struct stat st{};
    Envelope e{};
    bool good=fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_uid==getuid() && !(st.st_mode&0077)
      && st.st_size==sizeof(e) && read(fd,&e,sizeof(e))==sizeof(e);
    close(fd);
    if(!good || std::memcmp(e.epoch,epoch.data(),64))return false;
    const Packet& p=e.command;
    if(std::memcmp(p.magic,"GRTGRIP1",8) || !std::isfinite(p.source) || !std::isfinite(p.opening)
      || !std::isfinite(p.expiry) || p.opening<.002/.044 || p.opening>.042/.044
      || p.expiry<p.source || p.expiry-p.source>.400000001 || p.source>time+1e-8 || p.expiry<time)return false;
    if(active && p.sequence==last.sequence)return std::memcmp(&p,&last,sizeof(p))==0;
    const double width=.044*p.opening;
    if(time-p.source>.025 || (active && (p.sequence!=last.sequence+1 || p.source<=last.source
      || std::abs(width-.044*last.opening)>.04*(p.source-last.source)+1e-12))) return false;
    if(!active){
      if(p.sequence!=0 || std::abs(target[0]-width)>1e-8 || std::abs(target[1]-width)>1e-8)return false;
      event("COMMAND_EPOCH_RESET",tick,target,measured);
    }
    last=p;active=true;event("RECEIVER_ACK",tick,target,measured);
    return true;
  }
  void stop(const char* reason,uint64_t tick,const double* target,const double* measured){
    if(!stopped){stopped=true;event(reason,tick,target,measured);}
  }
  void observed(uint64_t tick,const double* target,const double* measured){
    event("NATIVE_STEP",tick,target,measured);
    if(tick==stop_tick)stop("RECEIVER_STOP_AT_CAP",tick,target,measured);
  }
};
}  // namespace student_grip
