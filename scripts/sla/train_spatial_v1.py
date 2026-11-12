"""Finite matched S2 qualification or fixed500; no scheduler or physical owner."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from go2arx5_vla.sla_adapter.spatial_runtime_v1 import SpatialEngine, common_optimizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from scripts.sla.train_v1 import probes

ACTIVE=None


def common_snapshot(engine,pin):
    """Compare streamed common values to a separately sealed S1 saved state."""
    import jax
    import jax.numpy as jnp
    root=Path(pin['path'])
    if sha(root/'COMPLETE.json')!=pin['sha256']:
        raise ValueError('S1_COMMON_REFERENCE_COMPLETE_PIN_CHANGED')
    m=json.loads((root/'COMPLETE.json').read_bytes())
    for key in ('parent_manifest','tokenizer','target_audit','visit_map_sha256','lr_sha256','seed',
                'frozen_fingerprint','upstream_sha','codec_version','initial_function_rule','XLA_FLAGS','optimizer'):
        if m['bindings'][key]!=engine.bindings[key]:
            raise ValueError('S1_S2_COMMON_RECIPE_IDENTITY_CHANGED:'+key)
    for name,value in m['bindings']['adapter_source'].items():
        if engine.bindings['adapter_source'].get(name)!=value:
            raise ValueError('S2_CHANGED_COMMON_S1_IMPLEMENTATION:'+name)
    tree=(engine.active,common_optimizer(engine.opt_state),engine.rng)
    leaves,_=jax.tree_util.tree_flatten_with_path(tree)
    if m['step']!=engine.step or m['cursor']!=engine.cursor or len(leaves)!=len(m['arrays']):
        raise ValueError('MATCHED_COMMON_STEP_CURSOR_COMPLETE_STATE_REQUIRED')
    exact=jax.jit(jnp.array_equal)
    for (key,current),row in zip(leaves,m['arrays']):
        file=root/row['file']
        if (not file.resolve().is_relative_to(root.resolve()) or sha(file)!=row['sha256']
            or jax.tree_util.keystr(key)!=row['key'] or list(current.shape)!=row['shape']
            or str(current.dtype)!=row['dtype']):
            raise ValueError('S1_COMMON_SAVED_STATE_ARRAY_IDENTITY_MISMATCH')
        reference=np.load(file,allow_pickle=False)
        if not bool(exact(current,jax.device_put(reference))):
            raise ValueError('S2_COMMON_STATE_NOT_BIT_EXACT_TO_S1_SAVED_STATE:'+row['key'])
        del reference
    return dict(reference=pin,step=engine.step,cursor=engine.cursor,arrays=len(leaves),bit_exact=True)


def same_initial_probes(rows,pin):
    reference=read_pin(pin)
    keys=('index','noise','normalized_sha256','physical_sha256')
    if len(rows)!=16 or len(reference)!=16 or any(any(a[k]!=b[k] for k in keys) for a,b in zip(rows,reference)):
        raise ValueError('S2_DOES_NOT_SHARE_S1_INITIAL_ORDINARY_RGB_FUNCTION')
    return dict(reference=pin,comparisons=16,all_bit_exact=True)


def run(spec):
    import jax
    op=spec['operation']
    allowed=('QUALIFY_SPATIAL_ALPHA0','QUALIFY_SPATIAL_CONTINUOUS','QUALIFY_SPATIAL_SPLIT',
             'QUALIFY_SPATIAL_RESUME','CANDIDATE_SPATIAL')
    if op not in allowed:
        raise ValueError('ONE_FINITE_SPATIAL_OPERATION_REQUIRED')
    if op=='CANDIDATE_SPATIAL':
        base=read_pin(spec['training_admission']);q=read_pin(spec['spatial_admission'])
        if (base['schema']!='SLA_CUDA_TRAIN_RESTORE_RGB_ADMISSION_V1' or not base['passes']
            or base['noncandidate_updates']!=8 or q['schema']!='SLA_SPATIAL_GRADIENT_CONTROL_ADMISSION_V1'
            or not q['passes'] or not q['alpha0_all_common_gradients_and_updates_bit_exact']
            or not q['positive_spatial_gradient_reaches_deployed_VLM_LoRA']
            or q['positive_noncandidate_updates']!=8 or q['runtime_source_sha']!=spec['source_sha']
            or q['teacher_cache']!=spec['teacher_cache'] or q['projector']!=spec['projector']):
            raise ValueError('REAL_MATCHED_SPATIAL_GRADIENT_RESTORE_RGB_ADMISSION_REQUIRED')
    global ACTIVE
    engine=ACTIVE=SpatialEngine.__new__(SpatialEngine)
    SpatialEngine.__init__(engine,spec,Path(spec['registered_root'])/'work')
    if spec.get('resume'):
        engine.resume(spec['resume'])
    stop=500 if op=='CANDIDATE_SPATIAL' else 2 if op in('QUALIFY_SPATIAL_ALPHA0','QUALIFY_SPATIAL_SPLIT') else 4
    result=dict(operation=op,noncandidate=op!='CANDIDATE_SPATIAL',requested_updates=stop-engine.step,
                requested_physical_requests=0,started_epoch=time.time(),alpha=engine.alpha)
    if engine.step==0:
        result['matched_common_initial_state']=common_snapshot(engine,spec['S1_common_step0'])
        if op in('QUALIFY_SPATIAL_CONTINUOUS','CANDIDATE_SPATIAL'):
            result['initial_RGB_probes']=probes(engine,'step0-RGB')
            result['matched_S1_initial_function']=same_initial_probes(result['initial_RGB_probes'],spec['S1_initial_RGB'])
        engine.checkpoint('step000')
    while engine.step<stop:
        if time.time()>=spec['deadline_epoch']:
            raise TimeoutError('FINITE_SPATIAL_TRAINING_DEADLINE')
        row=engine.update()
        if engine.step<=4 or engine.step%50==0:
            values=[v for v in row['micro']if 'loss'in v]
            print(json.dumps(dict(event='ACTUAL_MATCHED_SPATIAL_BATCH_UPDATE',step=engine.step,cursor=engine.cursor,
                FM_mean=sum(v['loss']for v in values)/32,spatial_mean=sum(v['spatial_loss']for v in values)/32,
                alpha=engine.alpha,grad_norm=row['grad_norm_before_clip'],wall_s=row['wall_s'])),flush=True)
        if engine.step==2 and op=='QUALIFY_SPATIAL_ALPHA0':
            result['matched_S1_after_two_updates']=common_snapshot(engine,spec['S1_common_step2'])
        if engine.step in ((250,500)if op=='CANDIDATE_SPATIAL'else(2,4)):
            result['last_checkpoint']=engine.checkpoint(f'step{engine.step:03d}')
            if op=='CANDIDATE_SPATIAL'and engine.step==250:
                result['midpoint_RGB_probes']=probes(engine,'step250-RGB')
    if op in('QUALIFY_SPATIAL_CONTINUOUS','CANDIDATE_SPATIAL'):
        result['endpoint_RGB_probes']=probes(engine,f'step{engine.step}-RGB')
    if not all(np.isfinite(np.asarray(v)).all()for v in jax.tree.leaves((engine.active,engine.projector))):
        raise ValueError('FINITE_S2_COMMON_AND_PROJECTOR_PARAMETERS_REQUIRED')
    result.update(status='COMPLETED_FINITE_SLA_OPERATION',step=engine.step,cursor=engine.cursor,
        actual=engine.counts,frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),
        load_wall_s=engine.load_wall_s,teacher_weights_loaded=False,completed_epoch=time.time())
    return result


if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    try:
        result=run(spec)
    except BaseException as e:
        failure=dict(status='TECHNICAL_SPATIAL_FAILURE_NOT_POLICY_RESULT',error=repr(e),operation=spec['operation'],
            actual=getattr(ACTIVE,'counts',{}),step=getattr(ACTIVE,'step',0),cursor=getattr(ACTIVE,'cursor',0),
            physical_requests=0,epoch=time.time())
        (Path(spec['registered_root'])/'FAILED.private.json').write_text(json.dumps(failure,indent=2))
        raise
    (Path(spec['registered_root'])/'RESULT.private.json').write_text(json.dumps(result,indent=2))
