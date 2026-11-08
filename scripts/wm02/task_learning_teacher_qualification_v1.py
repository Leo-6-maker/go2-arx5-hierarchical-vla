"""Twenty fixed serial Teacher requests via the existing Journal and native worker.

No new scheduler, GPU work, retries, controller changes or training data grant.
The panel and actual parent scope are registered separately before invocation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from contextlib import closing


def run(input_path):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model import task_learning_grasp_v1 as task
    from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as panel_contract
    from go2arx5_vla.world_model import task_learning_scene_v1 as scenes
    from go2arx5_vla.world_model.pilot1_scope import validate as validate_scope
    inp=json.loads(input_path.read_text());stage=input_path.parent
    lease_path=stage/'control/LEASE.json';lease=json.loads(lease_path.read_text())
    if (lease.get('status')!='ACTIVE'or lease.get('run_uuid')!=stage.name
        or lease.get('input_sha256')!=digest(input_path)or os.uname().nodename!='corelab-4080s'or os.getuid()!=2009):
        raise PermissionError('FRESH_ACTUAL_FINITE_TEACHER_PANEL_REQUIRED')
    panel=json.loads(Path(inp['panel']).read_text())
    if digest(inp['panel'])!=inp['panel_sha256']:raise ValueError('FROZEN_TWENTY_REQUEST_PANEL')
    rows=panel_contract.validate(panel);base=json.loads(Path(inp['base_spec']).read_text())
    if digest(inp['base_spec'])!=inp['base_spec_sha256']:raise ValueError('FROZEN_NATIVE_CONSTRUCTION_BASE')
    prior=Path(inp['base_spec']).parent
    if (json.loads((prior/'control/LEASE.json').read_text())['status']!='CLOSED'
        or json.loads((prior/'WORKER_EXIT.private.json').read_text())['exit_code']!=0
        or json.loads((prior/'native-01/TASK_LEARNING_SCORE.private.json').read_text())['task_success']is not True):
        raise ValueError('ACTUAL_COMPLETE_CONSTRUCTION_MUST_PRECEDE_QUALIFICATION')
    if (stage/'SUPERVISOR.private.json').exists():raise FileExistsError('RECONCILE_EXISTING_TEACHER_PANEL_NEVER_REPLAY')
    source=Path(inp['source_root']);campaign=Path(base['campaign_root'])
    validate_scope(dict(base,registered_root=str(stage),source_sha=inp['source_sha'],deadline_epoch=inp['deadline_epoch']))
    journal=Journal(campaign/'control/journal',campaign.name)
    start=time.monotonic();outcomes=[];code=1;failure=None
    write(stage/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=inp['source_sha'],started_epoch=time.time()))
    def interrupted(sig,frame):raise KeyboardInterrupt('TEACHER_PANEL_SIGNAL_'+str(sig))
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}

    def environment(root):
        env=dict(os.environ,**base['native_environment'])
        env.update(PYTHONPATH=str(source/'src')+':'+str(source)+':'+env.get('PYTHONPATH',''),
            CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',TMPDIR=str(root/'tmp'),ROS_LOG_DIR=str(root/'logs'),
            XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache/config'))
        return env

    def register(row):
        root=stage/'runs'/row['native_run'];key=root.name.upper()
        with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(base['physical_owner_lock'])):
            if root.exists()or journal.path(key+'-REGISTER').exists():raise FileExistsError('UNKNOWN_OR_EXISTING_QUALIFICATION_CASE')
            def effect():
                descriptor=inp['scenes'][row['layout']];proof=json.loads(Path(descriptor['path']).read_text())
                if digest(descriptor['path'])!=descriptor['sha256']:raise ValueError('FROZEN_QUALIFICATION_LAYOUT')
                if (proof['parent']!=row['parent']or proof['layout']!=row['layout']
                    or proof['compiled']['scene_sha256']!=row['model_sha256']):raise ValueError('PREREGISTERED_SCENE_ROW_JOIN')
                root.mkdir()
                for name in('control','cache','logs','data','tmp'):(root/name).mkdir()
                s=dict(base)
                s.update(source_root=str(source),source_sha=inp['source_sha'],source_files=inp['source_files'],
                    registered_root=str(root),role=key,lease=str(root/'control/LEASE.json'),
                    registry_snapshot=str(root/'control/REGISTRY_SNAPSHOT.private.json'),
                    deadline_epoch=min(time.time()+1200,inp['deadline_epoch']),wall_limit_s=1200,
                    task_learning_role=task.QUALIFICATION_ROLE,task_learning_grasp=task.protocol(task.QUALIFICATION_ROLE),
                    experiment_role='TASK_LEARNING_TEACHER_QUALIFICATION_NO_BC',
                    task_condition='FROZEN_V7_TEACHER_QUALIFICATION_20_INDEPENDENT_REQUESTS',
                    task_learning_scene_reference=descriptor,scene_id=row['parent'],data_parent=row['parent'],
                    model_path=str(Path(descriptor['path']).parent/'SCENE.private.mjb'),
                    stable_task_panel=inp['panel'],stable_task_panel_sha256=inp['panel_sha256'],
                    cpus=[0,1,2,3],ROS_DOMAIN_ID=220)
                s.pop('task_learning_control',None)
                for name in('cache','logs','data','tmp'):s[name]=str(root/name)
                s['native_environment']=dict(base['native_environment']);s['native_environment']['PYTHONPATH']=environment(root)['PYTHONPATH']
                write(root/'EPISODE_MANIFEST.private.json',scenes_spec_episode(s))
                s.update(episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
                    episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),
                    stage_identity_sha256=digest(root/'EPISODE_MANIFEST.private.json'),
                    scene_allowlist=inp['scene_allowlist'],scene_allowlist_sha256=inp['scene_allowlist_sha256'])
                write(root/'EXECUTION_SPEC.json',s)
                program="import json,sys;from go2arx5_vla.world_model.task_learning_grasp_native_v1 import episode_config;from go2arx5_vla.world_model.auto1_native import phase_limit;s=json.load(sys.stdin);episode_config(s);assert phase_limit(s['phase'],json.loads(open(s['previous_stage']).read()),s)==21203;assert not({'torch','mujoco','rclpy'}&set(sys.modules));print('PASS_QUALIFICATION_STATIC_INTERFACE_NO_NATIVE_START')"
                checked=subprocess.run([s['python'],'-B','-c',program],input=json.dumps(s),env=environment(root),cwd=source,capture_output=True,text=True,timeout=120)
                write(root/'STRUCTURAL_PREFLIGHT.private.json',dict(exit_code=checked.returncode,stdout=checked.stdout,stderr=checked.stderr[-5000:]))
                if checked.returncode:raise ValueError('QUALIFICATION_PREFLIGHT_FAILED:'+checked.stderr[-2000:])
                reg,pin=snapshot(base['registry'])
                if any(e.get('path')==str(root)for e in reg['entries']):raise ValueError('EXISTING_REGISTRY_CASE')
                reg['entries'].append(dict(role=key,path=str(root),run_uuid=root.name,scope='WM02_PILOT1_SIM_ONLY',
                    source_sha=inp['source_sha'],owner=base['owner'],executor_node=base['host'],
                    pilot1_binding=base['pilot1_binding'],local_lease=s['lease']))
                once(Path(base['registry']),reg,key+'-REGISTRY',expected_before_sha256=pin)
                write(s['registry_snapshot'],reg)
                write(s['lease'],dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(root/'EXECUTION_SPEC.json'),
                    owner=base['owner'],scope='WM02_PILOT1_SIM_ONLY',deadline_epoch=s['deadline_epoch']))
                validate_scope(s,reg)
                return dict(spec=str(root/'EXECUTION_SPEC.json'),spec_sha256=digest(root/'EXECUTION_SPEC.json'))
            return journal.once(key+'-REGISTER',dict(row=row,source_sha=inp['source_sha']),effect)

    def execute(registered):
        path=Path(registered['spec']);s=json.loads(path.read_text());root=path.parent
        if digest(path)!=registered['spec_sha256']:raise ValueError('FROZEN_REGISTERED_QUALIFICATION_SPEC')
        def effect():
            argv=[s['python'],'-B','-m','scripts.wm02.pilot1_native_worker','--spec',str(path)]
            with(root/'logs/worker-out.log').open('xb')as out,(root/'logs/worker-err.log').open('xb')as err:
                child=subprocess.Popen(argv,cwd=source,env=environment(root),stdout=out,stderr=err)
                try:
                    while child.poll()is None:
                        write(stage/'PROGRESS.private.json',dict(current_run=root.name,pid=child.pid,completed=len(outcomes),updated_epoch=time.time()))
                        if time.time()>=s['deadline_epoch']or(stage/'STOP.json').exists():
                            write(root/'STOP.json',dict(reason='FINITE_TEACHER_QUALIFICATION_ENDPOINT'))
                        time.sleep(.5)
                finally:
                    if child.poll()is None:
                        write(root/'STOP.json',dict(reason='OWNED_TEACHER_COORDINATOR_EXIT'))
                        try:child.wait(timeout=90)
                        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=60)
            return dict(exit_code=child.returncode)
        return journal.once(root.name.upper()+'-RUN',registered,effect)

    def observe(row):
        root=stage/'runs'/row['native_run'];native=root/'native-01'
        def read(path):return json.loads(path.read_text())if path.exists()else{}
        n=read(native/'RESULT.json');w=read(root/'WORKER_EXIT.private.json');l=read(root/'control/LEASE.json')
        ready=read(native/'TASK_LEARNING_START_READINESS.private.json');score=read(native/'TASK_LEARNING_SCORE.private.json')
        result=panel_contract.classify(row,native=n,worker=w,lease=l,readiness=ready,score=score,
            spec=read(root/'EXECUTION_SPEC.json'),expected_source=inp['source_sha'],attempted=native.exists())
        return dict(result,result_sha256=digest(native/'RESULT.json')if(native/'RESULT.json').exists()else None)

    def scenes_spec_episode(s):
        from go2arx5_vla.world_model.stable_task_scene_v1 import episode
        return episode(s)

    try:
        for row in rows:
            if time.time()+1200>inp['deadline_epoch']or(stage/'STOP.json').exists():
                raise RuntimeError('FINITE_PANEL_NO_LATE_NEW_EPISODE; REMAINING_REQUESTS_NOT_RUN')
            registered=register(row);execute(registered);result=observe(row)
            outcomes.append(result);write(stage/'OUTCOMES.private.json',outcomes)
            write(stage/'SUMMARY.private.json',panel_contract.summarize(panel,outcomes))
            if result['hard_safety_stop']:raise RuntimeError('ORIGINAL_HARD_SAFETY_STOP_PANEL_FAMILY_PAUSED')
            if not result['technical_valid']:raise RuntimeError('UNRESOLVED_ENGINEERING_FAILURE_NO_BLIND_NEXT_REQUEST')
        code=0
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        summary=panel_contract.summarize(panel,outcomes);write(stage/'SUMMARY.private.json',summary)
        write(stage/'RESULT.private.json',dict(status='TEACHER_QUALIFICATION_COMPLETE'if code==0 else'TEACHER_QUALIFICATION_INCOMPLETE_PRESERVED',
            exit_code=code,failure=failure,summary=summary,wall_s=time.monotonic()-start,
            CPU_s=sum(v['CPU_s']for v in outcomes),qualified_training_demonstrations=0))
        current=json.loads(lease_path.read_text());current.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=code);write(lease_path,current)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    return run(p.parse_args(argv).input)


if __name__=='__main__':raise SystemExit(main())
