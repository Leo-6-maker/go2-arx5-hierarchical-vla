"""Isolated exact-step pause plumbing; frozen controller/policy math is reused.

This compiles a copy of the already-qualified clock adapter, never the vendor
tree. Paused FSM/leg history do not consume wall-time inference waits.
"""
import difflib
import json
import os
from pathlib import Path
import shlex
import subprocess
from .task_capture_runtime import cpu_stage, read, sha


CPP = r'''
void RL_Sim::PauseTaskClock(uint64_t step) {
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);
    sim->run = 0;
    simulation_running = false;
    task_clock_paused_ = true;
    task_clock_step_ = step;
    PI0294SimNs.store(static_cast<int64_t>(std::llround(mj_data->time * 1e9)));
    if (const char* p = std::getenv("GR00T_TASK_PAUSE_ACK")) {
        std::string temp = std::string(p) + ".pending";
        std::ofstream f(temp);
        f << std::setprecision(17) << "{\"step\":" << step
          << ",\"time_s\":" << mj_data->time
          << ",\"motiontime\":" << motiontime
          << ",\"leg_history_steps\":" << episode_length_buf
          << ",\"gait_indices\":" << gait_indices
          << ",\"ack_sequence\":" << ++task_clock_ack_sequence_
          << ",\"running\":false,\"clock_paused\":true}\n";
        f.close();
        if (!f || std::rename(temp.c_str(), p)) std::abort();
    } else std::abort();
}
extern "C" void GR00TTaskPause(uint64_t step) {
    if (!RL_Sim::instance) std::abort();
    RL_Sim::instance->PauseTaskClock(step);
}
'''


def patch(cpp, hpp, utils):
    """Assertions bound each change to one known scheduling location."""
    needle = '    static RL_Sim* instance;'
    assert hpp.count(needle) == 1
    hpp = hpp.replace(needle, needle+'\n    void PauseTaskClock(uint64_t step);\n    bool task_clock_paused_ = false;\n    uint64_t task_clock_step_ = 0, task_clock_ack_sequence_ = 0;')
    needle = '    if (params.Get<bool>("native_mrt", false)) native_lock.lock();'
    assert cpp.count(needle) == 1
    cpp = cpp.replace(needle, '    native_lock.lock(); // Atomic pause against leg inference/history.')
    needle = '    this->StateController(&this->robot_state, &this->robot_command);'
    assert cpp.count(needle) == 1
    # Original initial paused startup is unchanged. Only a quota pause takes
    # this branch; it publishes actual state without running the FSM again.
    pause = '''    if (task_clock_paused_) {
        const bool resume = this->control.current_keyboard == Input::Keyboard::Enter;
        if (!resume) {
            PauseTaskClock(task_clock_step_);
#ifdef USE_MUJOCO_ROS2
            PublishRosArmState();
            PublishRosOdometry();
#endif
            return;
        }
        task_clock_paused_ = false;
    }
'''
    cpp = cpp.replace(needle, pause+needle)
    cpp += '\n'+CPP
    needle = 'while (Seconds((d->time - syncSim)*slowdown)'
    assert utils.count(needle) == 1
    utils = utils.replace(needle, 'while (sim.run && Seconds((d->time - syncSim)*slowdown)')
    return cpp, hpp, utils


def run(root, native, startup):
    from scripts.pi0292.durable import write
    root, native, startup = map(Path, (root, native, startup))
    with cpu_stage(root, 'TASK_PAUSE_CLOCK_BUILD_01', cores=4, cpu_s_cap=1800):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise PermissionError('CPU_ONLY')
        parent = Path(read(root/'OUTPUT_BINDING.private.json')['parent_root'])
        old = parent/'fullbody/clock-v1'; old_receipt = read(old/'CLOCK_BUILD_RECEIPT.json')
        for p, h in old_receipt['source_pins'].items():
            if sha(native/p) != h:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED')
        if sha(old/'rl_sim_mujoco.cpp') != '211750eed952f48820c08214e356f3f77a0690a833c5a24d909be587a7ed4cc7':
            raise ValueError('QUALIFIED_CLOCK_BASE_CHANGED')
        out = root/'runtime/TASK_PAUSE_CLOCK_01'; out.mkdir(exist_ok=False)
        names = ('rl_sim_mujoco.cpp','rl_sim_mujoco.hpp','mujoco_utils.hpp')
        before = [(old/n).read_text() for n in names]
        after = patch(*before)
        for n, text in zip(names, after):(out/n).write_text(text)
        for n in ('native_home.hpp','native_startup.hpp'):(out/n).write_bytes((old/n).read_bytes())
        (out/'PRIVATE_TIMING.diff').write_text(''.join(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=n,tofile=n)) for n,a,b in zip(names,before,after)))
        flags={};directory=native/'build/mujoco';cmake=directory/'CMakeFiles/rl_sim_mujoco.dir'
        for line in (cmake/'flags.make').read_text().splitlines():
            if ' = ' in line:
                k,v=line.split(' = ',1);flags[k]=shlex.split(v)
        commands=[]
        def execute(argv,label,cwd=None):
            p=subprocess.run(argv,cwd=cwd,text=True,capture_output=True,timeout=600)
            (out/(label+'.log')).write_text(p.stdout+p.stderr)
            commands.append(dict(label=label,argv=argv,exit=p.returncode))
            write(out/'BUILD_COMMANDS.private.json',commands);p.check_returncode()
        obj=out/'rl_sim_mujoco.cpp.o'
        execute(['/usr/bin/c++','-I'+str(out),'-isystem','/opt/ros/jazzy/include/rosgraph_msgs']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-o',str(obj),'-c',str(out/names[0])], 'compile')
        link=shlex.split((cmake/'link.txt').read_text())
        link[link.index('CMakeFiles/rl_sim_mujoco.dir/src/rl_sim_mujoco.cpp.o')]=str(obj)
        link[link.index('CMakeFiles/rl_sim_mujoco.dir/library/thirdparty/mujoco_simulate/simulate.cc.o')]=str(startup/'startup-v2/simulate.cc.o')
        link.append('/opt/ros/jazzy/lib/librosgraph_msgs__rosidl_typesupport_cpp.so')
        link[link.index('-o')+1]=str(out/'rl_sim_mujoco');execute(link,'link',directory)
        deps=subprocess.run(['ldd',str(out/'rl_sim_mujoco')],text=True,capture_output=True)
        (out/'ldd.log').write_text(deps.stdout+deps.stderr)
        if deps.returncode or 'not found' in deps.stdout:raise ValueError('TIMING_LIBRARY_UNRESOLVED')
        symbols=subprocess.check_output(['nm','-D',str(out/'rl_sim_mujoco')],text=True)
        if not any(s.endswith(' T GR00TTaskPause') for s in symbols.splitlines()):raise ValueError('PAUSE_EXPORT_UNRESOLVED')
        # The original solver, node, static libraries and policies are untouched.
        receipt=dict(status='TIMING_OVERLAY_COMPILED_CPU_ONLY',source_pins=old_receipt['source_pins'],
            original_clock_binaries=old_receipt['binary_sha256'],binary_sha256={'rl_sim_mujoco':sha(out/'rl_sim_mujoco')},
            sources={n:sha(out/n) for n in names},private_diff_sha256=sha(out/'PRIVATE_TIMING.diff'),
            controller_math_changes=0,gain_changes=0,safety_threshold_changes=0,physics_changes=0,
            pause_scope='ACTUAL_STEP_QUOTA_PHYSICS_FSM_AND_LEG_HISTORY',new_physics=0,new_neural_queries=0)
        write(out/'CLOCK_BUILD_RECEIPT.json',receipt);print(json.dumps(receipt))
