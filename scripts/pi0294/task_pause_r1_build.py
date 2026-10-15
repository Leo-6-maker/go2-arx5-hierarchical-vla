"""Correct task-owned pause synchronization without serializing normal N1 PD.

The prior task adapter broadened the vendor's conditional sim mutex to all
policies. Restore that exact conditional; use a separate lifecycle mutex for
atomic pause/history. Native policy, periods, physics and control math stay put.
"""
import difflib
import os
from pathlib import Path
import shlex
import subprocess
from .task_pause_build import patch as previous_patch
from .night_r1_runtime import read,sha
from scripts.pi0292.durable import write


def patch(cpp,hpp,utils):
    original=(cpp,hpp,utils)
    cpp,hpp,utils=previous_patch(cpp,hpp,utils)
    bad='    native_lock.lock(); // Atomic pause against leg inference/history.'
    assert cpp.count(bad)==1
    restored='    if (params.Get<bool>("native_mrt", false)) native_lock.lock();'
    cpp=cpp.replace(bad,restored+'\n    const std::lock_guard<std::recursive_mutex> pause_guard(task_model_pause_mutex_);')
    declaration='    bool task_clock_paused_ = false;'
    assert hpp.count(declaration)==1
    hpp=hpp.replace(declaration,'    std::recursive_mutex task_model_pause_mutex_;\n'+declaration)
    needle='    if (task_clock_paused_) {'
    assert cpp.count(needle)==1
    guard='''    std::unique_lock<std::recursive_mutex> lifecycle_guard(task_model_pause_mutex_, std::defer_lock);
    if (task_clock_paused_ || this->control.current_keyboard == Input::Keyboard::Enter ||
        this->control.current_gamepad == Input::Gamepad::RB_X) lifecycle_guard.lock();
'''
    cpp=cpp.replace(needle,guard+needle)
    needle='void RL_Sim::PauseTaskClock(uint64_t step) {\n    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);'
    assert cpp.count(needle)==1
    cpp=cpp.replace(needle,needle+'\n    const std::lock_guard<std::recursive_mutex> pause_guard(task_model_pause_mutex_);')
    # Verify the complete normal native actor computation after locking is
    # identical, not merely that a few numeric constants still match.
    def part(text,begin,end):return text.split(begin,1)[1].split(end,1)[0]
    assert part(original[0],'    if (this->rl_init_done && simulation_running)','std::vector<float> RL_Sim::Forward()')==part(cpp,'    if (this->rl_init_done && simulation_running)','std::vector<float> RL_Sim::Forward()')
    assert part(original[0],'void RL_Sim::SetCommand(', 'void RL_Sim::RobotControl()')==part(cpp,'void RL_Sim::SetCommand(', 'void RL_Sim::RobotControl()')
    return cpp,hpp,utils


HARNESS=r'''
#include <mutex>
#include <future>
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
int main() {
  std::recursive_mutex sim, lifecycle;
  std::promise<void> model_entered, release_model, pd_done;
  auto release=release_model.get_future().share();
  auto model=std::async(std::launch::async,[&]{std::lock_guard<std::recursive_mutex> g(lifecycle);model_entered.set_value();release.wait();});
  model_entered.get_future().wait();
  auto pd=std::async(std::launch::async,[&]{std::lock_guard<std::recursive_mutex> g(sim);pd_done.set_value();});
  assert(pd_done.get_future().wait_for(1s)==std::future_status::ready);
  std::promise<void> pause_started,pause_done;
  auto paused=pause_done.get_future();
  auto pause=std::async(std::launch::async,[&]{std::lock_guard<std::recursive_mutex> s(sim);pause_started.set_value();std::lock_guard<std::recursive_mutex> g(lifecycle);pause_done.set_value();});
  pause_started.get_future().wait();
  assert(paused.wait_for(0s)==std::future_status::timeout);
  release_model.set_value();assert(paused.wait_for(1s)==std::future_status::ready);
  model.get();pd.get();pause.get();
  // Nested RobotControl -> PauseTaskClock uses the same ordered lifecycle lock.
  {std::lock_guard<std::recursive_mutex> s(sim);std::lock_guard<std::recursive_mutex> g(lifecycle);std::lock_guard<std::recursive_mutex> again(lifecycle);}
}
'''


def run(root,native,startup):
    root,native,startup=map(Path,(root,native,startup))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY')
    manifest=read(root/'RUN_MANIFEST.json');old=Path(manifest['dependencies']['native_clock_root']);old_receipt=read(old/'CLOCK_BUILD_RECEIPT.json')
    for p,h in old_receipt['source_pins'].items():
        if sha(native/p)!=h:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED')
    assert sha(old/'rl_sim_mujoco.cpp')=='211750eed952f48820c08214e356f3f77a0690a833c5a24d909be587a7ed4cc7'
    out=root/'runtime/TASK_PAUSE_R1_02';out.mkdir(exist_ok=False)
    names=('rl_sim_mujoco.cpp','rl_sim_mujoco.hpp','mujoco_utils.hpp');before=[(old/n).read_text() for n in names];after=patch(*before)
    for n,text in zip(names,after):(out/n).write_text(text)
    for n in ('native_home.hpp','native_startup.hpp'):(out/n).write_bytes((old/n).read_bytes())
    (out/'PRIVATE_TIMING.diff').write_text(''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=n,tofile=n)) for n,a,b in zip(names,before,after)))
    commands=[]
    def execute(argv,label,cwd=None):
        p=subprocess.run(argv,cwd=cwd,text=True,capture_output=True,timeout=600)
        (out/(label+'.log')).write_text(p.stdout+p.stderr);commands.append(dict(label=label,argv=argv,exit=p.returncode));write(out/'BUILD_COMMANDS.private.json',commands);p.check_returncode()
    (out/'pause_harness.cpp').write_text(HARNESS)
    execute(['/usr/bin/c++','-std=c++17','-pthread',str(out/'pause_harness.cpp'),'-o',str(out/'pause_harness')],'compile_harness')
    execute([str(out/'pause_harness')],'run_harness')
    flags={};directory=native/'build/mujoco';cmake=directory/'CMakeFiles/rl_sim_mujoco.dir'
    for line in (cmake/'flags.make').read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);flags[k]=shlex.split(v)
    obj=out/'rl_sim_mujoco.cpp.o'
    execute(['/usr/bin/c++','-I'+str(out),'-isystem','/opt/ros/jazzy/include/rosgraph_msgs']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-o',str(obj),'-c',str(out/names[0])],'compile')
    link=shlex.split((cmake/'link.txt').read_text());link[link.index('CMakeFiles/rl_sim_mujoco.dir/src/rl_sim_mujoco.cpp.o')]=str(obj)
    link[link.index('CMakeFiles/rl_sim_mujoco.dir/library/thirdparty/mujoco_simulate/simulate.cc.o')]=str(startup/'startup-v2/simulate.cc.o')
    link.append('/opt/ros/jazzy/lib/librosgraph_msgs__rosidl_typesupport_cpp.so');link[link.index('-o')+1]=str(out/'rl_sim_mujoco');execute(link,'link',directory)
    deps=subprocess.run(['ldd',str(out/'rl_sim_mujoco')],text=True,capture_output=True);(out/'ldd.log').write_text(deps.stdout+deps.stderr)
    if deps.returncode or 'not found' in deps.stdout:raise ValueError('TIMING_LIBRARY_UNRESOLVED')
    symbols=subprocess.check_output(['nm','-D',str(out/'rl_sim_mujoco')],text=True)
    if not any(s.endswith(' T GR00TTaskPause') for s in symbols.splitlines()):raise ValueError('PAUSE_EXPORT_UNRESOLVED')
    receipt=dict(status='PASS_CPU_PAUSE_SYNCHRONIZATION_BUILD_ONLY',source=manifest['code']['vla_commit'],
        source_pins=old_receipt['source_pins'],original_clock_binaries=old_receipt['binary_sha256'],binary_sha256={'rl_sim_mujoco':sha(out/'rl_sim_mujoco')},
        sources={n:sha(out/n) for n in names},private_diff_sha256=sha(out/'PRIVATE_TIMING.diff'),
        normal_native_N1_sim_mutex_conditional_restored=True,full_native_RunModel_computation_equal=True,full_native_SetCommand_equal=True,
        normal_PD_progress_during_inflight_inference_CPU_PASS=True,pause_waits_for_inflight_inference_CPU_PASS=True,nested_pause_lock_CPU_PASS=True,
        native_period_changes=0,controller_math_changes=0,gain_changes=0,safety_threshold_changes=0,physics_changes=0,
        old_pause_binary_preserved=True,new_physics=0,new_neural_queries=0)
    write(out/'CLOCK_BUILD_RECEIPT.json',receipt);print('R1_PAUSE_BUILD='+__import__('json').dumps(receipt));return receipt
