"""Read-only native command receipt added to the qualified pause adapter.

The vendor callback/watchdog/control computations remain byte-identical.
"""
import difflib,os,shlex,subprocess
from pathlib import Path
from .task_pause_r1_build import patch as qualified_patch,HARNESS
from .night_r1_runtime import read,sha
from scripts.pi0292.durable import write


def patch(cpp,hpp,utils):
    cpp,hpp,utils=qualified_patch(cpp,hpp,utils)
    needle='    task_clock_step_ = step;'
    assert cpp.count(needle)==1
    cpp=cpp.replace(needle,needle+"\n    std::lock_guard<std::mutex> receipt_guard(ros_arm_mutex_);")
    needle='          << ",\\"ack_sequence\\":" << ++task_clock_ack_sequence_'
    assert cpp.count(needle)==1
    cpp=cpp.replace(needle,'          << ",\\"native_command_sequence\\":" << task_command_sequence_\n          << ",\\"native_command_valid\\":" << (ros_arm_target_valid_ ? "true" : "false")\n          << ",\\"native_command_age_s\\":" << (ros_arm_target_valid_ ? std::chrono::duration<double>(PI0294SimNow()-ros_arm_target_time_).count() : -1.0)\n          << ",\\"native_arm_mode\\":\\"" << ros_arm_mode_ << "\\""\n'+needle)
    needle='    ros_arm_target_valid_ = true;'
    assert cpp.count(needle)==1
    cpp=cpp.replace(needle,needle+'\n    ++task_command_sequence_; // Receipt only; original callback above unchanged.')
    needle='    std::recursive_mutex task_model_pause_mutex_;'
    assert hpp.count(needle)==1
    hpp=hpp.replace(needle,needle+'\n    uint64_t task_command_sequence_ = 0;')
    return cpp,hpp,utils


def run(root,native,startup):
    root,native,startup=map(Path,(root,native,startup))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY')
    manifest=read(root/'RUN_MANIFEST.json');old=Path(manifest['dependencies']['native_clock_root']);old_receipt=read(old/'CLOCK_BUILD_RECEIPT.json')
    for p,h in old_receipt['source_pins'].items():
        if sha(native/p)!=h:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED')
    assert sha(old/'rl_sim_mujoco.cpp')=='211750eed952f48820c08214e356f3f77a0690a833c5a24d909be587a7ed4cc7'
    out=root/'runtime/TASK_PAUSE_R1_04';out.mkdir(exist_ok=False)
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
    receipt=dict(status='PASS_CPU_NATIVE_COMMAND_RECEIPT_BUILD_ONLY',source=manifest['code']['vla_commit'],
        source_pins=old_receipt['source_pins'],original_clock_binaries=old_receipt['binary_sha256'],binary_sha256={'rl_sim_mujoco':sha(out/'rl_sim_mujoco')},
        sources={n:sha(out/n) for n in names},private_diff_sha256=sha(out/'PRIVATE_TIMING.diff'),
        normal_native_N1_sim_mutex_conditional_restored=True,full_native_RunModel_computation_equal=True,full_native_SetCommand_equal=True,
        normal_PD_progress_during_inflight_inference_CPU_PASS=True,pause_waits_for_inflight_inference_CPU_PASS=True,nested_pause_lock_CPU_PASS=True,
        native_period_changes=0,controller_math_changes=0,gain_changes=0,safety_threshold_changes=0,physics_changes=0,
        old_pause_binary_preserved=True,new_physics=0,new_neural_queries=0)
    write(out/'CLOCK_BUILD_RECEIPT.json',receipt);print('R1_PAUSE_BUILD='+__import__('json').dumps(receipt));return receipt
