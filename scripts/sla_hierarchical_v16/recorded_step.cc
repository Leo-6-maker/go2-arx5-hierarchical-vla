// Test-only next-step implementation. Replay measured states without integrating dynamics.
#include <mujoco/mujoco.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
std::uint64_t count = 0;
template <class T> T read_field(const unsigned char* p) {
  T x;
  std::memcpy(&x, p, sizeof x);
  return x;
}
}

extern "C" std::uint64_t V3RecordedCount() { return count; }

extern "C" void mj_step(const mjModel* model, mjData* data) {
  static FILE* archive = std::fopen(std::getenv("V3_REPLAY_JOURNAL"), "rb");
  static FILE* calls = std::fopen("delivered_ticks.bin", "wbx");
  std::uint32_t n = 0;
  if (!archive || !calls || std::fread(&n, sizeof n, 1, archive) != 1 ||
      n < 36 + 8 * (34 + 32 + 20) || n > (1u << 20)) std::abort();
  std::vector<unsigned char> frame(n);
  if (std::fread(frame.data(), 1, n, archive) != n) std::abort();
  const auto* p = frame.data();
  const auto tick = read_field<std::uint64_t>(p + 8);
  if (std::memcmp(p, "GRTSTATE", 8) || tick != count + 1 ||
      model->nq != 34 || model->nv != 32 || model->nu != 20 ||
      read_field<std::uint32_t>(p+24) != 34 || read_field<std::uint32_t>(p+28) != 32 ||
      read_field<std::uint32_t>(p+32) != 20) std::abort();
  std::memcpy(data->qpos,p+36,8*34);
  std::memcpy(data->qvel,p+36+8*34,8*32);
  data->time=read_field<double>(p+16);
  // Recompute geometric contacts for the receiver's normal observation code.
  mj_forward(model,data);
  if (std::fwrite(&tick,sizeof tick,1,calls)!=1 || std::fflush(calls)) std::abort();
  count=tick;
}
