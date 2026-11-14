"""Finite build-only callback replay candidate;never launches a native world."""
import json,os,shutil,subprocess,time
from pathlib import Path

from go2arx5_vla.world_model.storage import digest,write_json
from go2arx5_vla.world_model.native_callback_replay_v61 import VERSION,patch,write_schedule


def build(spec,output):
    output=Path(output);started=time.monotonic()
    if (spec['schema']!=VERSION+'_BUILD_SPEC' or spec['physics']is not False
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists()):
        raise ValueError('NEW_FINITE_CPU_BUILD_NOT_NATIVE_EXECUTION')
    source=Path(spec['source_root']);oldspec=Path(spec['original_spec']['path'])
    if digest(oldspec)!=spec['original_spec']['sha256']:raise ValueError('ORIGINAL_NATIVE_SPEC_BYTES')
    original=json.loads(oldspec.read_bytes())
    producer=Path(original['registered_root'])
    lease=json.loads((producer/'control/LEASE.json').read_bytes())
    if lease['status']!='CLOSED' or lease['worker_exit_code']!=0:raise ValueError('CLOSED_ORIGINAL_EXPERT_PRODUCER')
    from go2arx5_vla.world_model.native_ui_rate_v1 import baseline_spec
    baseline_spec(original)
    ui=Path(original['native_ui_build']['path']).parent;ui_spec=json.loads((ui.parent/'EXECUTION_SPEC.json').read_bytes())
    base=Path(ui_spec['baseline_candidate'])
    # Trusted build artifacts are checked once here,not in each native callback.
    for name,pin in ui_spec['frozen_files'].items():
        if digest(name)!=pin:raise ValueError('FROZEN_REUSED_BUILD_INPUT_CHANGED:'+name)
    if digest(producer/'native-01/init.trace')!=spec['original_trace']['sha256']:
        raise ValueError('EXACT_ORIGINAL_CALLBACK_TRACE')
    old=json.loads((base/'COMMANDS.private.json').read_bytes())
    main=[r['argv']for r in old if r['label']=='main_compile'and r['exit_code']==0]
    commands=json.loads((ui/'COMMANDS.private.json').read_bytes())
    links=[r['argv']for r in commands if r['label']=='native_link'and r['exit_code']==0]
    if len(main)!=1 or len(links)!=1:raise ValueError('ONE_ORIGINAL_COMPILE_AND_LINK')
    compile_args=list(main[0]);link=list(links[0]);cpp=base/'rl_sim_mujoco.cpp';utils=base/'mujoco_utils.hpp'
    if compile_args[compile_args.index('-c')+1]!=str(cpp) or link.count(str(base/'rl_sim_mujoco.cpp.o'))!=1:
        raise ValueError('ONLY_FROZEN_MAIN_OBJECT_REPLACED')
    a,b=patch(cpp.read_text(),utils.read_text());output.mkdir()
    for name in('rl_sim_mujoco.hpp','native_home.hpp','native_startup.hpp','init_trace.hpp',
                 'control_clock.hpp','applied_control_clock.hpp','command_pair.hpp','state_snapshot.hpp'):
        shutil.copyfile(base/name,output/name)
    shutil.copyfile(source/'src/go2arx5_vla/target_mujoco/callback_replay_v61.hpp',output/'callback_replay_v61.hpp')
    (output/'rl_sim_mujoco.cpp').write_text(a);(output/'mujoco_utils.hpp').write_text(b)
    schedule=write_schedule(producer/'native-01/init.trace',output/'recorded-prefix',stop_tick=5703)
    compile_args[compile_args.index('-c')+1]=str(output/'rl_sim_mujoco.cpp')
    compile_args[compile_args.index('-o')+1]=str(output/'rl_sim_mujoco.cpp.o')
    compile_args.insert(1,'-I'+str(output))
    link[link.index(str(base/'rl_sim_mujoco.cpp.o'))]=str(output/'rl_sim_mujoco.cpp.o')
    link[link.index('-o')+1]=str(output/'rl_sim_mujoco')
    actual=[]
    for label,argv in [('main_compile',compile_args),('native_link',link),
            ('native_ldd',['ldd',str(output/'rl_sim_mujoco')]),('native_needed',['readelf','-d',str(output/'rl_sim_mujoco')])]:
        if time.time()>spec['deadline_epoch']-60:raise TimeoutError('FINITE_REPLAY_BUILD_CLOSEOUT')
        q=subprocess.run(argv,cwd=ui_spec['native_build_directory'],text=True,capture_output=True,
                         timeout=min(900,max(1,spec['deadline_epoch']-time.time()-60)))
        (output/(label+'.log')).write_text(q.stdout+q.stderr)
        actual.append(dict(label=label,argv=argv,exit_code=q.returncode))
        write_json(output/'COMMANDS.private.json',actual);q.check_returncode()
    if 'not found'in(output/'native_ldd.log').read_text():raise ValueError('REPLAY_CANDIDATE_DEPENDENCIES')
    for name,pin in ui_spec['frozen_files'].items():
        if digest(name)!=pin:raise ValueError('ORIGINAL_INPUT_CHANGED_DURING_BUILD')
    result=dict(schema=VERSION+'_BUILD_ACTUAL',status='COMPILED_NOT_NATIVE_QUALIFIED',
        source_sha=spec['source_sha'],binary_sha256=digest(output/'rl_sim_mujoco'),
        baseline_binary_sha256=digest(original['native_binary']),original_spec=spec['original_spec'],
        original_trace=spec['original_trace'],schedule_sha256=schedule['schedule_sha256'],
        recorded_callbacks=schedule['callbacks'],recorded_prefix_end_tick=5703,
        original_cpp_sha256=digest(cpp),original_utils_sha256=digest(utils),
        patched_cpp_sha256=digest(output/'rl_sim_mujoco.cpp'),
        patched_utils_sha256=digest(output/'mujoco_utils.hpp'),
        replay_header_sha256=digest(output/'callback_replay_v61.hpp'),
        original_control_and_physics_text_exact_after_removing_gates=True,
        original_SDK_archive_reused_unchanged=True,original_solver_controller_viewer_reused=True,
        WBC_weights_gains_and_periods_changed=False,physics_dt_changed=False,
        model_loads=0,optimizer_updates=0,physical_requests=0,physics_integrations=0,
        native_state_reconstruction_proven=False,wall_s=time.monotonic()-started)
    write_json(output/'RESULT.json',result);return result


if __name__=='__main__':
    import sys
    s=json.loads(Path(sys.argv[1]).read_bytes());print(json.dumps(build(s,Path(s['registered_root'])/'candidate-01')))
