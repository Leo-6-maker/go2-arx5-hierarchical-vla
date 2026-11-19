"""One clean-base PD4 fit, fixed500 endpoint, full restore and32 RGB probes.

Uses the existing V6 reader, model, FM32 gradient and DeviceResidentEngine
update. No native simulator, Teacher, joint loss, smoothing or RP8 gate.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.pd4_training_v71 import OPERATION,MANIFESTS,STATISTICS
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha

ACTIVE_ENGINE = None
STOP_REQUESTED = False


def admission(spec):
    from go2arx5_vla.sla_adapter.pd4_training_v71 import OPERATION,MANIFESTS,native_qualification
    if (spec['operation']!=OPERATION or spec.get('resume') or spec.get('continuation')
            or spec['physics']is not False or spec['scope']!='SLA_PD4_MATCHED_OFFLINE_ONLY_V71'
            or spec['microbatch']!=8 or spec['training_matmul_precision']!='highest'
            or spec['dense_manifest_sha256']!=MANIFESTS[spec['PD4_view']]
            or (spec['planned_candidate_updates'],spec['planned_target_visits'],spec['planned_source_visits'])!=(500,16000,0)
            or spec['native_execution_admitted']is not False):
        raise ValueError('EXACT_MATCHED_PD4_500_FM32_FULL_TRUE_ONLY')
    native_qualification(spec['PD4_native_qualification'])


def prediction_metrics(dataset, index, normalized, physical):
    from scripts.sla.audit_saved_reference_probes_v61 import preview_metrics
    from go2arx5_vla.sla_adapter.codec_v1 import pad32
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, geodesic_angle
    ep, anchor = dataset.rows[index]
    data = dataset.labels[ep['id']]
    target = data['nominal_policy_H40'][anchor]
    expected = pad32(dataset.statistics['actions'].normalize(data['latent7'][anchor])).astype(np.float32)
    squared = (normalized[0].astype(float)-expected)**2
    metrics, failed = preview_metrics(physical, data['state_eef10'][anchor])
    return dict(episode=ep['id'], anchor=anchor, phase=str(data['phase'][anchor]),
        xyz_RMSE_m=float(np.sqrt(np.mean(np.sum((physical[:,:3]-target[:,:3])**2,axis=1)))),
        mean_SO3_rad=float(np.mean(geodesic_angle(rot6d_rows_to_matrix(physical[:,3:9]),rot6d_rows_to_matrix(target[:,3:9])))),
        nominal_grip_MAE=float(np.mean(abs(physical[:,9]-target[:,9]))),
        generation_MSE32=float(squared.mean()), generation_MSE7=float(squared[:,:7].mean()),
        generation_MSE25=float(squared[:,7:].mean()), EEF_full_H40_pass=not failed,
        failed_metrics=failed, geometric_metrics=metrics,
        joint_native_qualification='NOT_ESTABLISHED_BY_EEF_PROBES', full_WBC_state_restored=False)


def verify_restored(engine, checkpoint, *, expected_arrays=87):
    import jax
    manifest = json.loads((Path(checkpoint['path'])/'COMPLETE.json').read_bytes())
    leaves, _ = jax.tree_util.tree_flatten_with_path((engine.active, engine.opt_state, engine.rng))
    if len(leaves) != len(manifest['arrays']) or (expected_arrays is not None and len(leaves) != expected_arrays):
        raise ValueError('COMPLETE_RESTORED_MODEL_ADAM_RNG_ARRAY_COUNT_MISMATCH')
    for (key, value), row in zip(leaves, manifest['arrays'], strict=True):
        path = Path(checkpoint['path'])/row['file']
        if (jax.tree_util.keystr(key) != row['key'] or sha(path) != row['sha256']
                or not np.array_equal(np.asarray(value), np.load(path,allow_pickle=False))):
            raise ValueError('ALL_RESTORED_MODEL_ADAM_RNG_ARRAYS_MUST_MATCH')
    return dict(bit_exact=True, arrays=len(leaves), step=engine.step, cursor=engine.cursor,
                model_Adam_RNG_restored=True, optimizer_updates_added=0)


def train(spec):
    admission(spec)
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site = validate(spec)
    from go2arx5_vla.sla_adapter.pd4_training_v71 import PD4Engine
    from scripts.sla.train_v1 import probes
    from go2arx5_vla.sla_adapter.provider_v3 import write_verified_package
    import jax, flax, optax
    root = Path(spec['registered_root'])
    local = dict(spec, deadline_epoch=spec['deadline_epoch']-spec['closeout_reserve_s'], gradient_capture_steps=[])
    global ACTIVE_ENGINE
    engine = ACTIVE_ENGINE = PD4Engine.__new__(PD4Engine)
    PD4Engine.__init__(engine, local, root/'work')
    if engine.initial_frozen!=read_pin(spec['original_N20_initial_complete'])['bindings']['frozen_fingerprint']:
        raise ValueError('MATCHED_ORIGINAL_DENSE_FROZEN_FINGERPRINT')
    engine.bindings['training_site'] = dict(GPU_uuid=site['GPU_uuid'],GPU_name=site['GPU_name'],
        jax=jax.__version__,flax=flax.__version__,optax=optax.__version__)
    checkpoint0 = engine.checkpoint('step0000')
    original = read_pin(spec['original_N20_initial_complete'])
    new = json.loads((Path(checkpoint0['path'])/'COMPLETE.json').read_bytes())
    before = {r['key']: r for r in original['arrays']}
    if len(before) != 87 or len(new['arrays']) != 87 or any(any(r[k] != before[r['key']][k] for k in ('shape','dtype','sha256')) for r in new['arrays']):
        raise ValueError('PD4_START_MUST_MATCH_ORIGINAL87_INITIAL_ARRAYS')
    (root/'COMMON_INITIAL_ACTUAL.private.json').write_text(json.dumps(dict(arrays=87,bit_exact=True,optimizer_updates=0)))
    checkpoints = {0: checkpoint0}
    try:
        while engine.step < 500:
            if STOP_REQUESTED or time.time() >= local['deadline_epoch']:
                checkpoint = engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(status='FINITE_PD4_PREFIX_SAVED_NOT500',exit_code=75,
                    step=engine.step,target_visits=engine.cursor,source_visits=0,
                    last_checkpoint=checkpoint,actual=engine.counts,physical_requests=0)
            try:
                row = engine.update()
            except TimeoutError:
                checkpoint = engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(status='FINITE_PD4_PREFIX_SAVED_NOT500',exit_code=75,
                    step=engine.step,target_visits=engine.cursor,source_visits=0,
                    last_checkpoint=checkpoint,actual=engine.counts,physical_requests=0)
            if not row['grad_norm_before_clip'] > 0:
                raise ValueError('REAL_NONZERO_PD4_GRADIENT_REQUIRED')
            if engine.step == 1 or engine.step % 25 == 0:
                print(json.dumps(dict(step=engine.step,cursor=engine.cursor,FM32=row['mean_FM32_loss'],
                    active7=row['active7'],padding25=row['padding25'],used_lr=row['used_lr'],wall_s=row['wall_s'])),flush=True)
            if engine.step in (100,500):
                checkpoints[engine.step] = engine.checkpoint(f'step{engine.step:04d}')
        engine.resume(checkpoints[500])
        restored = verify_restored(engine, checkpoints[500])
        records = probes(engine,'step500-RGB',indices=engine.selected_indices)
        detailed = []
        for row in records:
            with np.load(engine.output/'step500-RGB'/row['file'],allow_pickle=False) as p:
                detailed.append(dict(row,**prediction_metrics(engine.dataset,row['index'],p['normalized'],p['physical_H40'])))
        assert len(detailed) == 32
        (root/'PREDICTIONS32.private.json').write_text(json.dumps(detailed,indent=2,allow_nan=False))
        dataset = Path(spec['dataset']);m = engine.dataset.manifest
        statistics = dataset/m['statistics']['file']
        assert sha(statistics) == STATISTICS[spec['PD4_view']]
        package = root/'RGB_PACKAGE'
        complete = json.loads((Path(checkpoints[500]['path'])/'COMPLETE.json').read_bytes())
        assert complete['bindings']['recipe'] == OPERATION and complete['step'] == 500 and complete['cursor'] == 16000
        write_verified_package(package,checkpoints[500]['path'],complete=complete,active_template=engine.active,
            statistics=statistics,tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        fit = all(r['EEF_full_H40_pass'] and r['xyz_RMSE_m'] <= .002 and r['mean_SO3_rad'] <= .01
                  and r['nominal_grip_MAE'] <= .02 for r in detailed)
        return dict(schema='SLA_PD4_OFFLINE500_ACTUAL_V7',status='COMPLETE_FIXED500_OFFLINE_ENDPOINT',exit_code=0,
            operation=OPERATION,PD4_view=spec['PD4_view'],step=engine.step,target_visits=engine.cursor,source_visits=0,actual=engine.counts,
            checkpoints=checkpoints,last_checkpoint=checkpoints[500],full_restore=restored,initial87_bit_exact=True,
            package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),
            prediction_manifest=str(root/'PREDICTIONS32.private.json'),predictions=32,
            EEF_full_H40_pass=sum(r['EEF_full_H40_pass'] for r in detailed),fit_targets_met=fit,
            frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),
            native_execution_admitted=False,physical_requests=0,native_task_success=False,
            dataset_manifest_sha256=MANIFESTS[spec['PD4_view']],statistics_sha256=STATISTICS[spec['PD4_view']],site=site)
    finally:
        engine.close()


def replay(spec):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    validate(spec)
    from go2arx5_vla.sla_adapter.provider_pd4_v71 import Provider
    from go2arx5_vla.sla_adapter.pd4_training_v71 import PD4Engine
    from go2arx5_vla.sla_adapter.pd4_training_v71 import fixed_probes
    import hashlib
    provider = Provider(spec['package']['path'],manifest_sha256=spec['package']['sha256'],
        parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    dataset = PD4Engine.__new__(PD4Engine).dataset_for_spec(spec)
    indices = fixed_probes(dataset)
    reference = json.loads(Path(spec['reference_probes']).read_bytes())
    rows = []
    for index in indices:
        state,_,images,text = dataset.raw(index)
        frame = dataset.frames[dataset.identity(index)['episode_id']]['T_policy_from_mount']
        for noise in range(2):
            value = provider.predict(state,images,text,frame,noise_id=noise)
            rows.append(dict(index=index,noise=noise,
                normalized_sha256=hashlib.sha256(value['normalized'].tobytes()).hexdigest(),
                physical_sha256=hashlib.sha256(value['physical_H40'].tobytes()).hexdigest()))
    if len(rows) != 32 or any(any(a[k] != b[k] for k in a) for a,b in zip(rows,reference,strict=True)):
        raise ValueError('FRESH_ORDINARY_RGB32_MUST_MATCH_PD4_ENDPOINT')
    return dict(status='FRESH_ORDINARY_RGB32_BIT_EXACT',exit_code=0,actual_parent_loads=1,
        ordinary_RGB_generations=32,optimizer_updates=0,physical_requests=0,all_bit_exact=True,
        optimizer_loaded=False,teacher_loaded=False,records=rows)


if __name__ == '__main__':
    import signal
    def stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        if ACTIVE_ENGINE is not None and hasattr(ACTIVE_ENGINE,'spec'):
            ACTIVE_ENGINE.spec['deadline_epoch'] = time.time()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();spec=json.loads(args.spec.read_bytes());root=Path(spec['registered_root'])
    destination=root/('REPLAY_RESULT.private.json' if args.replay else 'RESULT.private.json')
    if destination.exists():raise FileExistsError('EXISTING_PD4_RESULT_NO_REPLAY')
    try:
        result=(replay if args.replay else train)(spec)
    except BaseException as error:
        from scripts.pi0292.durable import write
        write(root/'FAILED.private.json',dict(error=repr(error),actual=getattr(ACTIVE_ENGINE,'counts',{}),
            step=getattr(ACTIVE_ENGINE,'step',0),cursor=getattr(ACTIVE_ENGINE,'cursor',0),physical_requests=0))
        raise
    from scripts.pi0292.durable import write
    write(destination,result)
    raise SystemExit(result['exit_code'])
