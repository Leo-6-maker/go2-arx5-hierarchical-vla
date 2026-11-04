// Bounded multi-producer trace. Only its writer thread performs disk writes.
#include "init_trace.hpp"
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <memory>
#include <string>
namespace {
static_assert(std::atomic<uint64_t>::is_always_lock_free&&std::atomic<float>::is_always_lock_free,"Emit paths require lock-free atomics");
constexpr uint64_t CAP=65536,SLOTS=32,FLOATS=16384,QCAP=4096;
struct Raw {uint64_t ticket,ns;double sim;uint32_t kind,n;uint64_t obs,policy,pos,vel,token;double data[96];};
static_assert(sizeof(Raw)==840,"Versioned little-endian trace layout");
struct Cell {std::atomic<uint64_t> ready{0};Raw row{};};
struct Snapshot {std::atomic<uint64_t> ready{0},token{0};std::atomic<uint32_t> kind{0},n{0};std::atomic<float> data[FLOATS]{};};
struct Queue {uint64_t ids[QCAP]{};std::atomic<uint64_t> head{0},tail{0};};
uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
thread_local uint64_t inference=0;
struct Recorder {
 std::unique_ptr<Cell[]> cells;std::unique_ptr<Snapshot[]> snapshots;Queue queues[2];
 std::atomic<uint64_t> head{0},dropped{0},obs{0},policy{0},consumed[2]{},applied[2]{},snapshot_head{0};
 std::atomic<bool> finish{false},stopped{false};uint64_t written=0;int fd=-1,sfd=-1;std::thread writer;
 Recorder(){
  const char*p=std::getenv("WM02_INIT_TRACE");if(!p)return;
  cells.reset(new Cell[CAP]);snapshots.reset(new Snapshot[SLOTS]);
  fd=open(p,O_WRONLY|O_CREAT|O_EXCL,0600);std::string path=std::string(p)+".snapshots";sfd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0||sfd<0)std::abort();const char magic[8]={'W','M','0','2','I','T','0','1'};write_all(fd,magic,8);uint32_t sizes[2]={sizeof(Raw),CAP};write_all(fd,sizes,8);
  writer=std::thread([this]{drain();});
 }
 static void write_all(int f,const void*p,size_t n){auto b=static_cast<const char*>(p);while(n){auto k=write(f,b,n);if(k<=0)std::abort();b+=k;n-=k;}}
 void emit(uint32_t kind,double t,uint64_t token,const float*values,uint32_t n,const double*numbers=nullptr){
  if(fd<0)return;if(kind==wm02_init::STOP)stopped.store(true);
  uint64_t ticket=head.fetch_add(1);if(ticket>=CAP){dropped.fetch_add(1);return;}
  auto &c=cells[ticket];auto&r=c.row;r.ticket=ticket;r.ns=ns();r.sim=t;r.kind=kind;r.n=std::min(n,96u);r.obs=obs.load();
  r.policy=policy.load();r.pos=applied[0].load();r.vel=applied[1].load();r.token=token;
  if(n<=96){for(uint32_t i=0;i<n;++i)r.data[i]=numbers?numbers[i]:values[i];}
  else if(numbers){r.n=0;r.data[0]=-1;r.data[2]=n;dropped.fetch_add(1);}
  else if(!stopped.load()){
   auto generation=snapshot_head.fetch_add(1);auto slot=generation<4?generation:4+(generation-4)%(SLOTS-4);auto&s=snapshots[slot];s.ready.store(0);
   s.token=token;s.kind=kind;s.n=std::min<uint32_t>(n,FLOATS);for(uint32_t i=0;i<s.n.load();++i)s.data[i].store(values[i],std::memory_order_relaxed);s.ready.store(generation+1,std::memory_order_release);
   r.n=0;r.data[0]=slot;r.data[1]=generation+1;r.data[2]=n;
   if(n>FLOATS)dropped.fetch_add(1);
  }else{r.n=0;r.data[0]=-1;r.data[2]=n;}
  c.ready.store(ticket+1,std::memory_order_release);
 }
 void drain(){uint64_t tail=0,seen[SLOTS]{};do{
   auto limit=std::min<uint64_t>(head.load(),CAP);
   while(tail<limit&&cells[tail].ready.load(std::memory_order_acquire)==tail+1){write_all(fd,&cells[tail].row,sizeof(Raw));++tail;}
   for(uint64_t i=0;i<SLOTS;++i){auto&s=snapshots[i];uint64_t g=s.ready.load(std::memory_order_acquire);if(!g||g==seen[i])continue;
    // Only the inference producer writes snapshots. The reader validates the
    // generation after copying before committing a bounded payload.
    float payload[FLOATS];auto token=s.token.load();auto kind=s.kind.load(),n=s.n.load();for(uint32_t j=0;j<n;++j)payload[j]=s.data[j].load(std::memory_order_relaxed);if(s.ready.load(std::memory_order_acquire)!=g)continue;
    uint64_t h[5]={g,i,token,kind,n};write_all(sfd,h,sizeof(h));write_all(sfd,payload,n*sizeof(float));seen[i]=g;
   }
   // Producers may append after this iteration's limit snapshot. After finish,
   // compare with the final head, not that stale snapshot, before exiting.
   if(finish.load()&&tail>=std::min<uint64_t>(head.load(),CAP))break;std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }while(true);
  written=tail; // Read only after writer.join() in the destructor.
 }
 ~Recorder(){if(fd<0)return;finish.store(true);writer.join();fsync(fd);fsync(sfd);close(fd);close(sfd);
  if(const char*p=std::getenv("WM02_INIT_TRACE")){std::string path=std::string(p)+".summary";FILE*f=fopen(path.c_str(),"wx");if(f){fprintf(f,"{\"events\":%llu,\"written\":%llu,\"dropped\":%llu,\"stop_latched\":%s}\n",(unsigned long long)head.load(),(unsigned long long)written,(unsigned long long)dropped.load(),stopped.load()?"true":"false");fclose(f);}}
 }
};
Recorder& recorder(){static Recorder r;return r;}
// Allocate and open trace files before native main/physics/control starts.
struct PreMain {PreMain(){recorder();}} premain;
}
extern "C" uint64_t WM02InitBegin(double t){auto&r=recorder();inference=++r.policy;r.emit(wm02_init::INFERENCE_BEGIN,t,inference,nullptr,0);return inference;}
extern "C" uint64_t WM02InitCurrent(){return inference;}
extern "C" void WM02InitEmit(uint32_t kind,double t,uint64_t token,const float*v,uint32_t n){auto&r=recorder();if(kind==wm02_init::OBS)++r.obs;r.emit(kind,t,token,v,n);}
extern "C" void WM02InitNumeric(uint32_t kind,double t,uint64_t token,const double*v,uint32_t n){auto&r=recorder();if(kind==wm02_init::OBS)++r.obs;r.emit(kind,t,token,nullptr,n,v);}
extern "C" void WM02InitPublish(uint32_t channel,uint64_t version){auto&r=recorder();if(channel>1)return;auto&q=r.queues[channel];auto h=q.head.load();if(h-q.tail.load()>=QCAP){++r.dropped;return;}q.ids[h%QCAP]=version;q.head.store(h+1,std::memory_order_release);float c=channel;r.emit(wm02_init::PUBLISH,0,version,&c,1);}
extern "C" void WM02InitConsume(uint32_t channel){auto&r=recorder();if(channel>1)return;auto&q=r.queues[channel];auto t=q.tail.load();uint64_t v=0;if(t<q.head.load(std::memory_order_acquire)){v=q.ids[t%QCAP];q.tail.store(t+1);}else ++r.dropped;r.consumed[channel].store(v);float c=channel;r.emit(wm02_init::CONSUME,0,v,&c,1);}
extern "C" void WM02InitApplied(){auto&r=recorder();for(int i=0;i<2;++i)r.applied[i].store(r.consumed[i].load());r.emit(wm02_init::APPLY,0,0,nullptr,0);}
