"""Reduce optional viewer work in a new binary; retain all controller objects."""
import json,os,subprocess,time
from pathlib import Path
from .storage import digest,write_json
from .target_gr00t_profile_v1 import need
from .stable_task_scene_v1 import pinned

VERSION='NATIVE_SOFTWARE_UI_FPS5_V1'
BASELINE='82c135c1bf5bb093cac832554b953d0017ec33e35e584865d08ca14327f3aa29'
INCLUDES='// WM02_NATIVE_SOFTWARE_UI_FPS5_V1\n#include <chrono>\n#include <thread>\n'
LOOP='  // run event loop\n  while (!this->platform_ui->ShouldCloseWindow() && !this->exitrequest.load()) {\n'
THROTTLED='''  // run event loop
  auto wm02_next_ui_frame = mj::Simulate::Clock::now();
  while (!this->platform_ui->ShouldCloseWindow() && !this->exitrequest.load()) {
    // Viewer pacing only, before acquiring the simulation mutex.
    const auto wm02_ui_now = mj::Simulate::Clock::now();
    if (wm02_ui_now < wm02_next_ui_frame) {
      std::this_thread::sleep_until(wm02_next_ui_frame);
    }
    if (this->exitrequest.load()) break;
    wm02_next_ui_frame = mj::Simulate::Clock::now() + std::chrono::milliseconds(200);
'''


def patch(source):
    start=source.index('void Simulate::RenderLoop() {')
    end=source.index('// add state to history buffer',start)
    body=source[start:end]
    need(source.count(LOOP)==body.count(LOOP)==1 and 'WM02_NATIVE_SOFTWARE_UI_FPS5_V1'not in source,
        'EXACT_UNPATCHED_ORIGINAL_VIEWER_LOOP')
    changed=INCLUDES+source[:start]+body.replace(LOOP,THROTTLED,1)+source[end:]
    need(changed[len(INCLUDES):].replace(THROTTLED,LOOP,1)==source,
        'VIEWER_ONLY_PATCH_REVERSES_TO_EXACT_ORIGINAL_SOURCE')
    return changed


def build(spec,output):
    """Compile one viewer object, relink sealed control objects, never run it."""
    need(spec.get('native_ui_rate')==VERSION and spec.get('physics')is False
        and spec.get('domain')=='TARGET_INIT_BUILD_ONLY_V1'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='','FINITE_CPU_VIEWER_BUILD_ONLY')
    root=Path(spec['baseline_build_root']);base=Path(spec['baseline_candidate'])
    need(base==root/'candidate-01'and root.resolve()==root and base.resolve()==base,
        'CANONICAL_COMPLETED_NATIVE_BASELINE')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    need(lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==worker.get('exit_code')==0,
        'ACTUAL_CLOSED_ZERO_EXIT_NATIVE_BASELINE_BUILD')
    for name,pin in spec['frozen_files'].items():
        need(digest(name)==pin,'UNCHANGED_FROZEN_UI_BUILD_INPUT_'+name)
    need(digest(base/'rl_sim_mujoco')==BASELINE,'EXACT_QUALIFIED_SENSOR_SNAPSHOT_NATIVE_BINARY')
    old=pinned(spec['baseline_build_commands']);recipe=pinned(spec['original_ui_build_receipt'])
    need(Path(spec['baseline_build_commands']['path'])==base/'COMMANDS.private.json',
        'ACTUAL_BASELINE_LINK_COMMANDS')
    links=[r['argv']for r in old if r['label']=='native_link'and r['exit_code']==0]
    compiles=[r['argv']for r in recipe['commands']if r['label']=='compile-simulate.cc'and r['exit_code']==0]
    need(len(links)==len(compiles)==1,'ONE_ACTUAL_UI_COMPILE_AND_BASELINE_LINK')
    compile_argv=list(compiles[0]);link=list(links[0]);original=Path(spec['original_ui_source'])
    ui_object=Path(compile_argv[compile_argv.index('-o')+1])
    need(compile_argv[compile_argv.index('-c')+1]==str(original)
        and link.count(str(ui_object))==1 and link.count(str(base/'rl_sim_mujoco.cpp.o'))==1
        and link.count(str(base/'librl_sdk.a'))==1,'ONLY_VIEWER_OBJECT_REPLACED_IN_QUALIFIED_LINK')
    need(str(original)in spec['frozen_files']and str(ui_object)in spec['frozen_files']
        and all(str(base/n)in spec['frozen_files']for n in('rl_sim_mujoco.cpp.o','librl_sdk.a')),
        'ALL_REUSED_CONTROL_AND_VIEWER_INPUTS_PINNED')
    output=Path(output);started=time.monotonic();source=output/'simulate.cc';source.write_text(patch(original.read_text()))
    compile_argv[compile_argv.index('-c')+1]=str(source)
    compile_argv[compile_argv.index('-o')+1]=str(output/'simulate.cc.o')
    link[link.index(str(ui_object))]=str(output/'simulate.cc.o')
    link[link.index('-o')+1]=str(output/'rl_sim_mujoco')
    commands=[]
    for label,args,cwd in [('ui_compile',compile_argv,spec['native_build_directory']),
                           ('native_link',link,spec['native_build_directory']),
                           ('native_ldd',['ldd',str(output/'rl_sim_mujoco')],None),
                           ('native_needed',['readelf','-d',str(output/'rl_sim_mujoco')],None)]:
        q=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=600)
        (output/(label+'.log')).write_text(q.stdout+q.stderr)
        commands.append(dict(label=label,argv=args,exit_code=q.returncode))
        write_json(output/'COMMANDS.private.json',commands);q.check_returncode()
    need('not found'not in(output/'native_ldd.log').read_text(),'UI_CANDIDATE_DYNAMIC_DEPENDENCIES_RESOLVE')
    for name,pin in spec['frozen_files'].items():need(digest(name)==pin,'FROZEN_INPUT_PRESERVED_AFTER_UI_BUILD')
    result=dict(status='UI_FPS5_CANDIDATE_COMPILED_NOT_NATIVE_QUALIFIED',version=VERSION,
        baseline_native_sha256=BASELINE,binary_sha256=digest(output/'rl_sim_mujoco'),
        patched_ui_source_sha256=digest(source),original_ui_source_sha256=digest(original),
        original_controller_and_leg_SDK_objects_reused=True,viewer_fps=5,
        training_dual_RGB_capture_modified=False,control_math_or_period_or_startup_or_safety_modified=False,
        new_native_integrations=0,new_policy_updates=0,wall_s=time.monotonic()-started)
    write_json(output/'RESULT.json',result)
    return result
