// This harness runs only after checking both resolved library identities.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <csignal>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
volatile std::sig_atomic_t received_interrupts = 0;
void note_interrupt(int) { ++received_interrupts; }
}
extern "C" const char* WM02OuterNext();
extern "C" std::uint64_t V3RecordedCount();
extern "C" void GR00TTaskPause(std::uint64_t tick) {
  std::ofstream f("pause_calls.txt",std::ios::app); f<<tick<<'\n';
  if(!f)std::abort();
}

void write_command(std::uint64_t sequence,double time,double width,double expiry) {
  double fraction=width/.044;
  std::ofstream f("command.pending",std::ios::binary);
  f.write(std::getenv("V16_GRIP_EPOCH"),64);
  f.write("GRTGRIP1",8);
  f.write(reinterpret_cast<const char*>(&sequence),8);
  for(double v:{time,fraction,expiry}) f.write(reinterpret_cast<const char*>(&v),8);
  f.close();
  if(!f || std::rename("command.pending","command.bin"))std::abort();
}

int main(int argc,char** argv) {
  const bool returning = argc == 7 && std::string(argv[6]) == "RETURNING";
  if (argc != 6 && !returning) return 2;
  std::filesystem::path original(argv[1]),reader(argv[2]);
  Dl_info first{};
  const char* next=WM02OuterNext();
  if(!dladdr(reinterpret_cast<void*>(&mj_step),&first) || !next ||
      !std::filesystem::equivalent(first.dli_fname,original) ||
      !std::filesystem::equivalent(next,reader))return 3;
  const std::string method=argv[5];
  if(method!="VALID" && method!="EXPIRY" && method!="OVER_RATE") return 4;
  std::ofstream route("route.txt");route<<first.dli_fname<<'\n'<<next<<'\n';
  if (returning) {
    const char* expected_writer = std::getenv("V3_FROZEN_TRACE_WRITER");
    Dl_info trace{};
    void* symbol = dlsym(RTLD_DEFAULT, "WM02InitEmit");
    if (!expected_writer || !symbol || !dladdr(symbol, &trace) ||
        !std::filesystem::equivalent(trace.dli_fname, expected_writer)) return 14;
    route << trace.dli_fname << '\n';
  }
  route.close();
  if(!route)return 5;
  mjModel* model=mj_loadModel(argv[3],nullptr);
  if(!model || model->nq!=34 || model->nv!=32)return 6;
  mjData* data=mj_makeData(model);
  if(!data)return 7;
  std::ifstream reset(argv[4],std::ios::binary);
  reset.read(reinterpret_cast<char*>(data->qpos),8*34);
  reset.read(reinterpret_cast<char*>(data->qvel),8*32);
  if(!reset || reset.peek()!=EOF)return 8;
  mj_forward(model,data);
  std::signal(SIGINT, returning ? note_interrupt : SIG_DFL);
  for(int tick=0;tick<6453;++tick){
    if(tick==0 || (tick>=2503 && (tick-2503)%25==0)){
      std::ofstream quota("quota");quota<<(tick==0 ? 2503 : tick+25)<<'\n';quota.close();
      if(!quota)return 9;
      if(tick>=2503){
        const double expiry=(tick==6428 && method=="EXPIRY") ? (tick+10.5)*.002 : data->time+.399;
        write_command((tick-2503)/25,data->time,.042,expiry);
      }
    }
    if(tick==6439 && method=="OVER_RATE")
      write_command((6428-2503)/25+1,data->time,.040,data->time+.399);
    mj_step(model,data);
    if (returning && tick == 6439 && method != "VALID") {
      if (received_interrupts != 1 || V3RecordedCount() != 6439) return 11;
      for (int attempt = 0; attempt < 14; ++attempt) {
        mj_step(model, data);
        if (V3RecordedCount() != 6439) return 12;
      }
      break;
    }
    if(V3RecordedCount()!=static_cast<std::uint64_t>(tick+1))return 10;
  }
  if (returning) {
    std::ofstream counts("signals.txt");
    counts << received_interrupts << '\n';
    counts.close();
    if (!counts) return 13;
  }
  mj_deleteData(data);mj_deleteModel(model);
  return 0;
}
