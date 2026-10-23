// Current zero-integration IO companion, not a controller or prefix validator.
// The admitted standard-entry owner pins this build and the original paused
// binary. No WBC source/binary/parameter, qpos/qvel/ctrl or callback is edited.
//
// Original GR00TTaskPause holds the sim's recursive mutex across rename of its
// ACK. Interpose that public libc boundary to copy GRTSTATE UNDER that mutex;
// never guess a C++ object's layout or read live mjData from a polling thread.
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

static_assert(mjVERSION_HEADER == 327, "exact current MuJoCo 3.2.7 header");
static_assert(sizeof(mjtNum) == 8, "original GRTSTATE float64 ABI");

namespace {
constexpr const char* ROLE = "DEC1_FROZEN_TARGET_NATIVE_IO_QUALIFICATION_V1";
using Rename = int(*)(const char*, const char*);
Rename real_rename() {
  static auto fn = reinterpret_cast<Rename>(dlsym(RTLD_NEXT, "rename"));
  if (!fn) _exit(95);
  return fn;
}
[[noreturn]] void fail(const char* reason) {
  if (const char* p = std::getenv("GR00T_TASK_ERROR")) {
    std::ofstream f(p, std::ios::app); f << reason << '\n'; f.flush();
  }
  // Every integration entry rejects BEFORE its original function. Do not run
  // destructors that could access a partly torn down native controller.
  _exit(94);
}
const char* required(const char* name) {
  const char* p = std::getenv(name);
  if (!p || *p != '/') fail("ZERO_IO_REQUIRED_ABSOLUTE_OWNED_PATH");
  return p;
}
bool exists(const std::string& p) { return access(p.c_str(), F_OK) == 0; }
bool exact_zero(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  char b[3]{}; f.read(b, 3);
  return f.gcount() == 2 && b[0] == '0' && b[1] == '\n';
}
double epoch() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
struct State {
  std::mutex mutex;
  const mjModel* model = nullptr;
  mjData* data = nullptr;
  bool started = false, capture = false, paused = false;
  uint64_t snapshots = 0;
  std::string quota, request, stop, stopped, state, ack;
  double deadline;
  State() {
    const char* role = std::getenv("GR00T_ZERO_IO_ROLE");
    if (!role || std::strcmp(role, ROLE)) fail("ZERO_IO_CURRENT_ROLE_REQUIRED");
    using Version = int(*)();
    auto version = reinterpret_cast<Version>(dlsym(RTLD_NEXT, "mj_version"));
    if (!version || version() != mjVERSION_HEADER) fail("ZERO_IO_EXACT_RUNTIME_HEADER_ABI_REQUIRED");
    quota = required("GR00T_TASK_STEP_QUOTA"); state = required("GR00T_TASK_STATE");
    ack = required("GR00T_TASK_PAUSE_ACK"); request = required("GR00T_ZERO_IO_PAUSE_REQUEST");
    stop = required("GR00T_ZERO_IO_STOP"); stopped = required("GR00T_ZERO_IO_STOPPED");
    required("GR00T_TASK_ERROR");
    const char* raw = std::getenv("GR00T_ZERO_IO_DEADLINE_EPOCH");
    char* end = nullptr; deadline = raw ? std::strtod(raw, &end) : 0;
    const double now = epoch();
    if (!raw || !end || *end || !std::isfinite(deadline) || deadline <= now ||
        deadline > now + 1800 || deadline > 1792866596.) fail("ZERO_IO_BOUNDED_ORIGINAL_DEADLINE");
    if (!exact_zero(quota) || exists(state) || exists(ack) || exists(stopped) || exists(request) || exists(stop))
      fail("ZERO_IO_NEW_OWNER_AND_ZERO_QUOTA_ONLY");
  }
};
// Deliberately process-lifetime storage: no global-destructor/worker race.
State& state() { static State* s = new State; return *s; }
template<class T> void put(std::ostream& f, const T& value) {
  f.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
void finite(const mjtNum* values, size_t n) {
  for (size_t i = 0; i < n; ++i) if (!std::isfinite(values[i])) fail("ZERO_IO_NONFINITE_ACTUAL_STATE");
}
void snapshot_locked(State& s) {
  const auto* m = s.model; const auto* d = s.data;
  if (!m || !d || m->nq != 34 || m->nv != 32 || m->nu != 20 || d->time != 0 ||
      !std::isfinite(d->time) || d->ncon < 0 || d->ncon > 4096 || !exact_zero(s.quota))
    fail("ZERO_IO_ACTUAL_STATE_OR_ZERO_QUOTA_CHANGED");
  finite(d->qpos, 34); finite(d->qvel, 32); finite(d->ctrl, 20);
  std::ostringstream f(std::ios::out | std::ios::binary);
  f.write("GRTSTATE", 8); const uint64_t step = 0; put(f, step); put(f, d->time);
  const uint32_t nq = 34, nv = 32, nu = 20, nc = d->ncon;
  put(f, nq); put(f, nv); put(f, nu);
  f.write(reinterpret_cast<const char*>(d->qpos), 8*nq);
  f.write(reinterpret_cast<const char*>(d->qvel), 8*nv);
  f.write(reinterpret_cast<const char*>(d->ctrl), 8*nu); put(f, nc);
  for (int i = 0; i < d->ncon; ++i) {
    const auto& c = d->contact[i]; const uint32_t a = c.geom1, b = c.geom2;
    using ContactForce = void(*)(const mjModel*, const mjData*, int, mjtNum*);
    static auto contact_force = reinterpret_cast<ContactForce>(dlsym(RTLD_NEXT, "mj_contactForce"));
    if (!contact_force) fail("ZERO_IO_ORIGINAL_CONTACT_FORCE_ENTRY_MISSING");
    mjtNum force[6]; contact_force(m, d, i, force);
    if (!std::isfinite(c.dist)) fail("ZERO_IO_NONFINITE_CONTACT");
    finite(force, 6); finite(c.frame, 9); finite(c.pos, 3);
    put(f, a); put(f, b); put(f, c.dist);
    f.write(reinterpret_cast<const char*>(force), 48);
    f.write(reinterpret_cast<const char*>(c.frame), 72);
    f.write(reinterpret_cast<const char*>(c.pos), 24);
  }
  const auto raw = f.str(); const std::string temporary = s.state + ".pending";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.write(raw.data(), raw.size()); out.close();
  if (!out || real_rename()(temporary.c_str(), s.state.c_str())) fail("ZERO_IO_STATE_PUBLICATION_FAILED");
  ++s.snapshots;
}
void* observer(void*) {
  State& s = state();
  while (true) {
    if (epoch() >= s.deadline) fail("ZERO_IO_FINITE_DEADLINE_REACHED");
    if (!exact_zero(s.quota)) fail("ZERO_IO_QUOTA_CHANGED");
    if (exists(s.stop)) {
      // Wait for any ACK hook in progress, then prevent later captures before
      // acknowledging. The owner waits for this BEFORE native SIGINT/teardown.
      std::lock_guard<std::mutex> lock(s.mutex); s.capture = false;
      std::ofstream out(s.stopped, std::ios::out | std::ios::trunc);
      out << "{\"schema\":\"DEC1_ZERO_IO_OBSERVER_STOPPED_V1\",\"observer_stopped\":true,"
          << "\"physics_integrations\":0,\"original_pause_called\":" << (s.paused ? "true" : "false")
          << ",\"snapshots\":" << s.snapshots << "}\n";
      out.close(); if (!out) fail("ZERO_IO_STOPPED_PUBLICATION_FAILED");
      return nullptr;
    }
    if (!s.paused && exists(s.request)) {
      if (!exact_zero(s.request)) fail("ZERO_IO_PAUSE_REQUEST_ZERO_ONLY");
      using Pause = void(*)(uint64_t);
      auto pause = reinterpret_cast<Pause>(dlsym(RTLD_DEFAULT, "GR00TTaskPause"));
      if (!pause) fail("ZERO_IO_EXACT_FROZEN_PAUSE_ENTRY_MISSING");
      { std::lock_guard<std::mutex> lock(s.mutex); s.capture = true; }
      // No pointer mutex held while acquiring the ORIGINAL native sim mutex.
      // The rename hook below captures while that ORIGINAL mutex is held.
      pause(0);
      { std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.snapshots) fail("ZERO_IO_FROZEN_ACK_RENAME_CAPTURE_MISSING");
        s.paused = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}

extern "C" int rename(const char* from, const char* to) {
  const int result = real_rename()(from, to);
  const char* ack = std::getenv("GR00T_TASK_PAUSE_ACK");
  if (!result && ack && !std::strcmp(ack, to)) {
    State& s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (s.capture) {
      Dl_info caller{};
      struct stat executable{}, origin{};
      // The original compiler may inline PauseTaskClock into its C ABI
      // wrapper. Both are in the exact pinned main executable and both hold
      // the ORIGINAL mutex over rename. No other caller/library is accepted.
      if (!dladdr(__builtin_return_address(0), &caller) || !caller.dli_sname || !caller.dli_fname ||
          (std::strcmp(caller.dli_sname, "_ZN6RL_Sim14PauseTaskClockEm") &&
           std::strcmp(caller.dli_sname, "GR00TTaskPause")) ||
          stat("/proc/self/exe", &executable) || stat(caller.dli_fname, &origin) ||
          executable.st_dev != origin.st_dev || executable.st_ino != origin.st_ino)
        fail("ZERO_IO_ACK_NOT_FROM_FROZEN_PAUSE_METHOD");
      snapshot_locked(s);
    }
  }
  return result;
}

extern "C" void mj_forward(const mjModel* m, mjData* d) {
  using Forward = void(*)(const mjModel*, mjData*);
  static auto real = reinterpret_cast<Forward>(dlsym(RTLD_NEXT, "mj_forward"));
  if (!real) fail("ZERO_IO_REAL_FORWARD_MISSING");
  State& s = state(); real(m, d);
  if (m->nq != 34 || m->nv != 32 || m->nu != 20) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.data && (s.data != d || s.model != m)) fail("ZERO_IO_AMBIGUOUS_NATIVE_DATA_POINTER");
  s.data = d; s.model = m;
  if (!s.started) {
    s.started = true; pthread_t thread;
    if (pthread_create(&thread, nullptr, observer, nullptr)) fail("ZERO_IO_OBSERVER_THREAD_FAILED");
    pthread_detach(thread);
  }
}

extern "C" void mj_deleteData(mjData* d) {
  using Delete = void(*)(mjData*);
  static auto real = reinterpret_cast<Delete>(dlsym(RTLD_NEXT, "mj_deleteData"));
  State& s = state();
  { std::lock_guard<std::mutex> lock(s.mutex);
    if (s.data == d) {
      if (s.capture || !exists(s.stopped)) fail("ZERO_IO_DELETE_BEFORE_OBSERVER_STOPPED");
      s.data = nullptr;
    }
  }
  if (!real) fail("ZERO_IO_REAL_DELETE_MISSING"); real(d);
}
extern "C" void mj_deleteModel(mjModel* m) {
  using Delete = void(*)(mjModel*);
  static auto real = reinterpret_cast<Delete>(dlsym(RTLD_NEXT, "mj_deleteModel"));
  State& s = state();
  { std::lock_guard<std::mutex> lock(s.mutex);
    if (s.model == m) {
      if (s.capture || !exists(s.stopped)) fail("ZERO_IO_DELETE_BEFORE_OBSERVER_STOPPED");
      s.model = nullptr;
    }
  }
  if (!real) fail("ZERO_IO_REAL_DELETE_MISSING"); real(m);
}

// Independent hard cap: even an accidental ENTER or a caller bypassing the
// Python quota object cannot perform a native integration in this component.
extern "C" void mj_step(const mjModel*, mjData*) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_step"); }
extern "C" void mj_step1(const mjModel*, mjData*) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_step1"); }
extern "C" void mj_step2(const mjModel*, mjData*) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_step2"); }
extern "C" void mj_Euler(const mjModel*, mjData*) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_Euler"); }
extern "C" void mj_RungeKutta(const mjModel*, mjData*, int) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_RungeKutta"); }
extern "C" void mj_implicit(const mjModel*, mjData*) { fail("ZERO_IO_INTEGRATION_FORBIDDEN_mj_implicit"); }
