"""One fixed1000 mobile FM endpoint; restore, RGB export and fresh replay.

Prediction geometry is reported, never used to suppress the protected native
TRAIN8/DEV16 panel. No native integration takes place in this worker.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.mfree_snapshot_training_v1 import SnapshotEngine, RECIPE
from scripts.sla.train_pd4_matched_v71 import verify_restored, prediction_metrics

ACTIVE_ENGINE=None
STOP_REQUESTED=False


def admission(spec):
    if (spec['operation']!=RECIPE 
            or spec['physics']is not False or spec['microbatch']!=8 or spec['training_matmul_precision']!='highest'
            or not spec['parent_host_cast_v7']
            or (spec['planned_candidate_updates'],spec['planned_target_visits'],spec['planned_source_visits'])!=(1000,32000,0)):
        raise ValueError('SINGLE_FIXED1000_NEW_MOBILE_RAW_PARENT_FM32_RECIPE')


def _all_update_lines(spec,current):
    lines=[]
    for pin in spec.get('prior_update_logs',[]):
        if sha(pin['path'])!=pin['sha256']:raise ValueError('IMMUTABLE_RESUME_UPDATE_LOG')
        lines.extend(Path(pin['path']).read_text().splitlines())
    lines.extend(current.read_text().splitlines())
    steps=[json.loads(line)['step']for line in lines]
    if steps!=list(range(1,1001)):raise ValueError('EXACT_ONCE_COMMITTED1000_UPDATE_LOG')
    return lines


def diagnostics(engine, name):
    from scripts.sla.train_v1 import probes
    records=probes(engine,name,indices=engine.selected_indices); detailed=[]
    for row in records:
        with np.load(engine.output/name/row['file'],allow_pickle=False)as z:
            detailed.append(dict(row,**prediction_metrics(engine.dataset,row['index'],z['normalized'],z['physical_H40'])))
    write(engine.output/name/'DETAILED.private.json',detailed)
    return detailed


def train(spec):
    admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.provider_v3 import write_verified_package
    site=validate(spec);root=Path(spec['registered_root']);local=dict(spec,deadline_epoch=spec['deadline_epoch']-spec['closeout_reserve_s'],gradient_capture_steps=[])
    global ACTIVE_ENGINE
    engine=ACTIVE_ENGINE=SnapshotEngine.__new__(SnapshotEngine)
    SnapshotEngine.__init__(engine,local,root/'work')
    original=read_pin(spec['original_N20_initial_complete']);checkpoints={}
    try:
        checkpoint0=engine.checkpoint('step0000');checkpoints[0]=checkpoint0
        actual=json.loads((Path(checkpoint0['path'])/'COMPLETE.json').read_bytes())
        before={r['key']:r for r in original['arrays']}
        if (len(before)!=87 or len(actual['arrays'])!=87
                or engine.initial_frozen!=original['bindings']['frozen_fingerprint']
                or any(any(r[k]!=before[r['key']][k]for k in ('shape','dtype','sha256'))for r in actual['arrays'])):
            raise ValueError('SAME87_RAW_PARENT_COMMON_ADAPTER_ADAM_RNG_INITIAL_ARRAYS')
        write(root/'COMMON_INITIAL_ACTUAL.private.json',dict(arrays=87,bit_exact=True,optimizer_updates=0))
        if spec.get('resume_pin'):
            from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import resumable_checkpoint
            previous=read_pin(dict(path=str(Path(spec['resume_pin']['path'])/'COMPLETE.json'),sha256=spec['resume_pin']['sha256']))
            resumable_checkpoint(previous,spec['resume_last_committed_step'],resumes=spec['resume_count']-1)
            engine.resume(spec['resume_pin']);verify_restored(engine,spec['resume_pin'])
            write(root/'RESUMED87_ACTUAL.private.json',dict(step=engine.step,cursor=engine.cursor,arrays=87,bit_exact=True))
        while engine.step<1000:
            if STOP_REQUESTED or time.time()>=local['deadline_epoch']:
                checkpoint=engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(status='FINITE_PREFIX_SAVED_NOT1000',exit_code=75,step=engine.step,
                    target_visits=engine.cursor,source_visits=0,last_checkpoint=checkpoint,actual=engine.counts,physical_requests=0)
            try:row=engine.update()
            except TimeoutError:
                checkpoint=engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(status='FINITE_PREFIX_SAVED_NOT1000',exit_code=75,step=engine.step,
                    target_visits=engine.cursor,source_visits=0,last_checkpoint=checkpoint,actual=engine.counts,physical_requests=0)
            if not row['grad_norm_before_clip']>0:raise ValueError('REAL_NONZERO_MOBILE_GRADIENT_REQUIRED')
            if engine.step==1 or engine.step%25==0:
                print(json.dumps(dict(step=engine.step,cursor=engine.cursor,FM32=row['mean_FM32_loss'],
                    active7=row['active7'],padding25=row['padding25'],used_lr=row['used_lr'],wall_s=row['wall_s'])),flush=True)
                write(root/'PROGRESS.private.json',dict(step=engine.step,cursor=engine.cursor,actual=engine.counts,last_update=row,epoch=time.time()))
            if engine.step in (250,500,750,1000):
                checkpoints[engine.step]=engine.checkpoint(f'step{engine.step:04d}')
                if engine.step==500:diagnostics(engine,'step500-TRAIN')
        engine.resume(checkpoints[1000]);restored=verify_restored(engine,checkpoints[1000])
        detailed=diagnostics(engine,'step1000-TRAIN');write(root/'ENDPOINT_RGB.private.json',detailed)
        complete=json.loads((Path(checkpoints[1000]['path'])/'COMPLETE.json').read_bytes())
        if complete['bindings']['recipe']!=RECIPE or complete['step']!=1000 or complete['cursor']!=32000:
            raise ValueError('TRUE_FIXED1000_COMPLETE_ENDPOINT')
        package=root/'RGB_PACKAGE';statistics=Path(spec['dataset'])/engine.dataset.manifest['statistics']['file']
        write_verified_package(package,checkpoints[1000]['path'],complete=complete,active_template=engine.active,
            statistics=statistics,tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        exposure={};episodes={};unique=set();dropped=0
        for line in _all_update_lines(spec,engine.output/'updates.private.jsonl'):
            update=json.loads(line)
            for micro in update['micro']:
                for index,drop in zip(micro['indices'],micro['state_token_dropped'],strict=True):
                    ep,a=engine.dataset.rows[index];phase=ep['expert_stage_audit_only'][a]
                    exposure[phase]=exposure.get(phase,0)+1;episodes[ep['id']]=episodes.get(ep['id'],0)+1
                    unique.add(index);dropped+=bool(drop)
        if sum(exposure.values())!=32000:raise ValueError('EXACT_REAL_OPTIMIZATION_VISIT_EXPOSURE')
        return dict(schema='MFREE_FIXED1000_TRAIN_ACTUAL_V1',status='COMPLETE_FIXED1000_RGB_PACKAGE',exit_code=0,
            operation=RECIPE,snapshot_id=spec['snapshot_id'],step=engine.step,target_visits=engine.cursor,source_visits=0,actual=engine.counts,
            last_checkpoint=checkpoints[1000],checkpoints=checkpoints,full_restore=restored,initial87_bit_exact=True,
            package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),
            diagnostics=dict(ordinary_RGB_predictions=len(detailed),EEF_full_H40_pass=sum(r['EEF_full_H40_pass']for r in detailed),
                controls_native_panel_admission=False),actual_exposure=dict(phases=exposure,episodes=episodes,
                unique_windows=len(unique),state_tokens_dropped=dropped),
            frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),site=site,
            physical_requests=0,native_task_success=False,fresh_ordinary_RGB_restore_pending=True,
            dataset_id=engine.dataset.manifest['dataset_id'],dataset_manifest_sha256=engine.dataset.manifest_sha,
            statistics_sha256=engine.dataset.manifest['statistics']['sha256'],optional_successor_started=False)
    finally:engine.close()


def replay(spec):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.provider_mfree_snapshot_v1 import Provider
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.mfree_snapshot_training_v1 import SnapshotDataset
    validate(spec);root=Path(spec['registered_root']);result=read_pin(spec['training_result'])
    provider=Provider(result['package']['path'],manifest_sha256=result['package']['sha256'],
        parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    dataset=SnapshotDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=spec['frame_cache_bytes'])
    training_root=Path(spec.get('training_root',root))
    reference=json.loads((training_root/'ENDPOINT_RGB.private.json').read_bytes());rows=[]
    import hashlib
    for record in reference:
        index=record['index'];state,_,images,text=dataset.raw(index)
        frame=dataset.frames[dataset.identity(index)['episode_id']]['T_policy_from_mount']
        value=provider.predict(state,images,text,frame,noise_id=record['noise'])
        row=dict(index=index,noise=record['noise'],normalized_sha256=hashlib.sha256(value['normalized'].tobytes()).hexdigest(),
            physical_sha256=hashlib.sha256(value['physical_H40'].tobytes()).hexdigest())
        if any(row[k]!=record[k]for k in row):raise ValueError('FRESH_RGB_PACKAGE_MUST_REPRODUCE_RESTORED_ENDPOINT')
        rows.append(row)
    return dict(schema='MFREE_FRESH_ORDINARY_RGB_RESTORE_ACTUAL_V1',exit_code=0,
        all_bit_exact=True,ordinary_RGB_generations=len(rows),optimizer_updates=0,training_visits=0,
        physical_requests=0,teacher_calls=0,optimizer_loaded=False,training_cache_required=False,records=rows)


if __name__=='__main__':
    import signal,traceback
    def stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
        if ACTIVE_ENGINE is not None and hasattr(ACTIVE_ENGINE,'spec'):ACTIVE_ENGINE.spec['deadline_epoch']=time.time()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('spec');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_bytes());root=Path(spec['registered_root'])
    path=root/('REPLAY_RESULT.private.json'if args.replay else'RESULT.private.json')
    if path.exists():raise FileExistsError('EXISTING_MOBILE_TRAIN_ENDPOINT_NO_REPLAY')
    try:result=(replay if args.replay else train)(spec)
    except BaseException:
        result=dict(exit_code=1,status='TECHNICAL_FAILURE_RETAINED',failure=traceback.format_exc(),
            step=getattr(ACTIVE_ENGINE,'step',0),cursor=getattr(ACTIVE_ENGINE,'cursor',0),
            actual=getattr(ACTIVE_ENGINE,'counts',{}),physical_requests=0)
    write(path,result);raise SystemExit(result['exit_code'])
