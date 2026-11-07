"""Execute a finite preregistered single-slot FUSION calibration group.

This is a scientific cohort driver using the existing Journal and native worker,
not a dispatcher or permission issuer. Its input/root/lease must already have
been registered by the canonical owner. New references run single-slot first.
"""
import argparse, json, os, resource, subprocess, time
from contextlib import closing
from pathlib import Path


def run(input_path):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from go2arx5_vla.world_model.fusion_contract import PLAN
    from go2arx5_vla.world_model.fusion_scope import validate, SCOPE
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.fusion_corrector import ARMS
    inp=json.loads(input_path.read_text());stage=input_path.parent
    base=json.loads(Path(inp['base_spec']).read_text());campaign=Path(base['campaign_root'])
    source=Path(inp['source_root']);lease=json.loads((stage/'control/LEASE.json').read_text())
    if (lease.get('status')!='ACTIVE' or lease.get('input_sha256')!=digest(input_path)
            or inp.get('fusion_plan_sha')!=PLAN or inp.get('max_native_slots')!=1
            or not Path(__file__).resolve().is_relative_to(source.resolve())
            or time.time()+600>=inp['deadline_epoch']):
        raise PermissionError('FRESH_FINITE_REGISTERED_SINGLE_SLOT_FUSION_GROUP_REQUIRED')
    if(stage/'STARTED.private.json').exists():raise FileExistsError('EXISTING_FUSION_GROUP_MUST_BE_RECONCILED')
    for name,pin in inp['source_files'].items():
        if digest(source/name)!=pin:raise ValueError('IMMUTABLE_FUSION_SOURCE_CHANGED:'+name)
    manifest=Path(inp['panel']);plan=json.loads(manifest.read_text())
    if digest(manifest)!=inp['panel_sha256']:raise ValueError('FUSION_PREREGISTERED_PANEL_CHANGED')
    rows=plan['rows']
    if (len(rows)!=5 or rows[0]['purpose']!='SCENE_QUALIFICATION'
            or {r['method']for r in rows[1:]}!=set(ARMS)
            or any(r['split']!='TRAIN'for r in rows)
            or len({r['parent']for r in rows})!=1
            or any(r['inference_mode']!='CALIBRATION_NO_VLA'for r in rows)):
        raise ValueError('FIRST_GROUP_ONE_NEW_TRAIN_PARENT_FOUR_ARMS_NO_VLA_REQUIRED')
    validate(dict(base,registered_root=str(stage),source_sha=inp['source_sha'],deadline_epoch=inp['deadline_epoch']))
    write(stage/'STARTED.private.json',dict(pid=os.getpid(),source_sha=inp['source_sha'],started_epoch=time.time()))
    j=Journal(campaign/'control/journal',campaign.name);results=[];code=1;failure=None;proof=None
    started=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        catalog=json.loads(Path(base['scene_allowlist']).read_text())
        for row in rows:
            if time.time()+600>=inp['deadline_epoch']:raise TimeoutError('NO_TIME_FOR_NEXT_FINITE_FUSION_EPISODE')
            root=Path(inp['output_parent'])/row['run_id'];key=root.name.upper()
            scene=next(s for s in catalog['scenes']if s['scene_id']==row['scene_id'])
            def register_child():
                with closing(OwnerLock(campaign/'OWNER.lock')),closing(OwnerLock(base['physical_owner_lock'])):
                    if root.exists():raise FileExistsError('RECONCILE_EXISTING_FUSION_CHILD')
                    root.mkdir()
                    for n in ['control','data','cache','logs','tmp']:(root/n).mkdir()
                    episode=dict(row,run=str(root));episode.pop('run_id')
                    write(root/'EPISODE_MANIFEST.private.json',episode)
                    s=dict(base,registered_root=str(root),role='WM02_'+key.replace('-','_'),
                        registry_snapshot=str(root/'REGISTRY_SNAPSHOT.private.json'),lease=str(root/'control/LEASE.json'),
                        deadline_epoch=min(time.time()+600,inp['deadline_epoch']),source_root=str(source),
                        source_sha=inp['source_sha'],source_files=inp['source_files'],cpus=[0,1,2,3],ROS_DOMAIN_ID=220,
                        episode_manifest=str(root/'EPISODE_MANIFEST.private.json'),
                        episode_manifest_sha256=digest(root/'EPISODE_MANIFEST.private.json'),
                        model_path=scene['model_path'])
                    s['frozen_files']=dict(base['frozen_files']);s['frozen_files'].pop(base['model_path'],None)
                    s['frozen_files'][scene['model_path']]=scene['model_sha256']
                    if proof is not None:s.update(previous_stage=str(proof),previous_stage_sha256=digest(proof),scene_qualification=str(proof),scene_qualification_sha256=digest(proof))
                    for n in ['data','cache','logs','tmp']:s[n]=str(root/n)
                    write(root/'EXECUTION_SPEC.json',s);reg,pin=snapshot(base['registry'])
                    reg['entries'].append(dict(role=s['role'],path=str(root),run_uuid=root.name,
                        source_sha=inp['source_sha'],owner=s['owner'],executor_node=s['host'],scope=SCOPE,
                        pilot1_binding=s['pilot1_binding'],fusion_binding=s['fusion_binding'],local_lease=s['lease']))
                    once(Path(base['registry']),reg,key+'-REGISTER',expected_before_sha256=pin)
                    write(root/'REGISTRY_SNAPSHOT.private.json',reg)
                    write(Path(s['lease']),dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(root/'EXECUTION_SPEC.json')))
                    return dict(spec_sha256=digest(root/'EXECUTION_SPEC.json'))
            admission=j.once(key+'-SPEC',dict(panel_sha256=inp['panel_sha256'],row=row),register_child)
            s=json.loads((root/'EXECUTION_SPEC.json').read_text())
            env=dict(os.environ,**base['native_environment']);env.update(PYTHONPATH=str(source/'src')+':'+str(source)+':'+env.get('PYTHONPATH',''),
                CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                TMPDIR=str(root/'tmp'),ROS_LOG_DIR=str(root/'logs'),XDG_CACHE_HOME=str(root/'cache'),XDG_CONFIG_HOME=str(root/'cache'))
            argv=['taskset','-c','0-3',s['python'],'-B','-m','scripts.wm02.pilot1_native_worker','--spec',str(root/'EXECUTION_SPEC.json')]
            def execute():
                with(root/'logs/out.log').open('xb')as out,(root/'logs/err.log').open('xb')as err:
                    worker=subprocess.Popen(argv,cwd=source,env=env,stdout=out,stderr=err)
                    while worker.poll()is None:
                        write(stage/'PROGRESS.private.json',dict(current_run=root.name,pid=worker.pid,completed=len(results),updated_epoch=time.time()))
                        if time.time()>=s['deadline_epoch']:write(root/'STOP.json',dict(reason='FINITE_FUSION_ENDPOINT'))
                        time.sleep(.5)
                return dict(argv=argv,exit_code=worker.returncode)
            worker=j.once(key+'-RUN',admission,execute);native=root/'native-01';p=native/'RESULT.json'
            n=json.loads(p.read_text())if p.exists()else{};score=native/'FUSION_SCORE.private.json'
            value=dict(run=root.name,method=row['method'],purpose=row['purpose'],worker=worker,
                native_status=n.get('status'),native_ticks=n.get('actual_target_integrations',0),
                native_result_sha256=digest(p)if p.exists()else None,shutdown_clean=n.get('shutdown_clean'),
                score=json.loads(score.read_text())if score.exists()else None,
                score_sha256=digest(score)if score.exists()else None)
            results.append(value);write(stage/'RESULTS.private.json',results)
            if (worker['exit_code']or n.get('status')!='NATIVE_STAGE_PASSED'or not n.get('shutdown_clean')
                    or n.get('physics_stops')or n.get('actual_target_integrations')!=4103):
                raise RuntimeError('FUSION_NATIVE_FAILURE_PRESERVED_NO_SCORE_BASED_RETRY')
            if row['purpose']=='SCENE_QUALIFICATION':proof=p
        code=0
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result=dict(schema='WM02_FUSION_FIRST_ACTUAL_CALIBRATION_GROUP_V1',source_sha=inp['source_sha'],
            exit_code=code,failure=failure,results=results,planned_method_evaluations=4,
            actual_method_evaluations=sum(r['purpose']!='SCENE_QUALIFICATION'for r in results),
            actual_native_ticks=sum(r['native_ticks']for r in results),WM_updates=0,Actor_updates=0,VLA_updates=0,VLA_inferences=0,
            exact_fork=False,physical_parents=1,split='TRAIN',TEST_read=False,
            CALIBRATION_CORRECTION_GAIN='NOT_ESTABLISHED_SINGLE_PARENT_ONE_RESTART',
            REAL_VLA_INFERENCE_DONE=False,FROZEN_VLA_NATIVE_COMPARISON=False,TASK_TRANSFER_GAIN=False,
            wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu,
            child_CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        write(stage/'RESULT.private.json',result);lease.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time())
        write(stage/'control/LEASE.json',lease)
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    return run(parser.parse_args(argv).input)


if __name__=='__main__':raise SystemExit(main())
