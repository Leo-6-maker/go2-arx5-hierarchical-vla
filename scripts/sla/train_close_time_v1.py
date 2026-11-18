"""Finite robot-time representation fallback and ordinary-RGB reference restore."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

from scripts.pi0292.durable import write
from scripts.sla.close_goal_g1_v1 import width_metrics, wait_resource_owner
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
STOP_REQUESTED=False


def evaluate(engine,spec,*,archived_contexts=None):
    from scripts.sla.export_mfree_c20_v1 import _reference
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    bank=read(Path(spec['previous_campaign'])/'diagnostic/WINDOW48_PLAN.private.json')['windows']
    root=Path(spec['registered_root']);source_rows={};records=[]
    if archived_contexts is not None and (
            archived_contexts['fixed_bank_sha256'] != spec['fixed_TRAIN48']['sha256']
            or archived_contexts['C20_manifest_sha256'] != engine.dataset.manifest_sha
            or set(archived_contexts['windows']) != {str(row['index']) for row in bank}):
        raise ValueError('D4_PORTABLE_TRAIN48_CONTEXT_IDENTITY')
    for w in bank:
        index=w['index'];ep,anchor=engine.dataset.rows[index];state,expert,_,_=engine.dataset.raw(index)
        if archived_contexts is None:
            native_spec=read(Path(ep['native_result']['path']).parent/'EXECUTION_SPEC.json')
            if ep['id']not in source_rows:
                capture=read(ep['original_live_capture']['path']);source_rows[ep['id']]=read(capture['labels']['path'])
            policy_from_world=native_spec['policy_from_world']
            reference_row=source_rows[ep['id']][anchor]
        else:
            context=archived_contexts['windows'][str(index)]
            if (context['episode'],context['anchor'])!=(ep['id'],anchor):
                raise ValueError('D4_EXACT_ARCHIVED_TRAIN48_EPISODE_AND_ANCHOR')
            policy_from_world=context['policy_from_world']
            reference_row=context
        for noise in (0,1):
            normalized,physical=engine.generate(index,noise)
            file=root/'data'/f'{index:04d}-{noise}.npz'
            with file.open('xb')as handle:np.savez(handle,normalized=normalized,physical_H40=physical)
            record=dict(**w,noise=noise,previous_reference=engine.dataset.reference(index),output=dict(path=str(file),sha256=sha(file)),
                normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                physical_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                width=width_metrics(physical,expert),
                envelope=analyze(physical,state,np.asarray(policy_from_world),_reference(reference_row)))
            records.append(record)
    write(root/'TRAIN48_PREDICTIONS.private.json',records)
    write(root/'ENDPOINT_RGB16.private.json',[records[2*j+n]for j in (0,4,8,12,16,20,28,36)for n in (0,1)])
    return dict(predictions=len(records),CLOSE_count=sum(r['phase']=='CLOSE'for r in records),
        CLOSE_pass=sum(r['phase']=='CLOSE'and r['envelope']['original_full_H40_pass']for r in records),
        nonCLOSE_count=sum(r['phase']!='CLOSE'for r in records),
        nonCLOSE_pass=sum(r['phase']!='CLOSE'and r['envelope']['original_full_H40_pass']for r in records),
        terminal_width_MAE_m=float(np.mean([r['width']['terminal_width_error_m']for r in records])),
        maximum_terminal_width_error_m=max(r['width']['terminal_width_error_m']for r in records),
        annotated_terminal_width_error_max_m=max(r['width']['terminal_width_error_m']for r in records if r['width']['expert_crossing_in_window']is not None),
        events=[dict(index=r['index'],noise=r['noise'],**r['width'])for r in records
                if r['width']['expert_crossing_in_window']is not None],
        physical_admission=False)


def train(spec,*,engine_class=None,export_package=None,initial_match_fn=None):
    import jax
    import jax.numpy as jnp
    import optax
    from go2arx5_vla.sla_adapter.time_training_v1 import TimeEngine as CloseEngine
    from go2arx5_vla.sla_adapter.batch_v2 import host_batch,device_batch
    from go2arx5_vla.sla_adapter.provider_time_v1 import export as write_verified_package
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from scripts.sla.train_pd4_matched_v71 import verify_restored
    if engine_class is not None:CloseEngine=engine_class
    if export_package is not None:write_verified_package=export_package
    root=Path(spec['registered_root']);started=time.time();site=validate(spec)
    engine=CloseEngine(spec,root/'work')
    try:
        # Diagnostic backward is charged separately, with no Adam mutation.
        host=host_batch(engine.dataset,engine.tokenizer,engine.visit_map,0,engine.microbatch)
        obs,actions=device_batch(host)
        noise,tau=engine.draw_bank(engine.rng,np.arange(32,dtype=np.uint32))
        (value,aux),gradient=engine.grad(engine.active,engine.frozen,obs,actions,noise[:engine.microbatch],tau[:engine.microbatch])
        norm=float(optax.global_norm(gradient));values=np.asarray([value,*aux])
        if not np.isfinite(values).all()or not np.isfinite(norm)or norm<=0:raise ValueError('REAL_DIAGNOSTIC_GRADIENT_NOT_QUALIFIED')
        write(root/'GRADIENT_ACTUAL.private.json',dict(optimizer_updates=0,optimization_visits=0,
            diagnostic_backward_calls=1,diagnostic_backward_windows=engine.microbatch,
            objective=float(values[0]),FM32=float(values[1]),active7=float(values[2]),padding25=float(values[3]),
            physical_width=float(values[4]),physical_delta=float(values[5]),gradient_norm=norm,
            noise_bank_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
            tau_bank_sha256=hashlib.sha256(np.asarray(tau).tobytes()).hexdigest(),
            visit_map_sha256=engine.bindings['visit_map_sha256'],lambda_value=spec['gripper_lambda']))
        del gradient
        initial=engine.checkpoint('step0000');checkpoints={0:initial}
        initial_complete=read(Path(initial['path'])/'COMPLETE.json')
        parent_manifest=read(Path(spec['C20_parent_package']['path'])/'MANIFEST.private.json')
        actual_active=[r for r in initial_complete['arrays']if r['key'].startswith('[0]')]
        if len(actual_active)!=len(parent_manifest['adapter_arrays'])or any(
            any(a[k]!=b[k]for k in ('key','shape','dtype','sha256'))
            for a,b in zip(actual_active,parent_manifest['adapter_arrays'])):
            raise ValueError('TRAINED_PARENT1000_FUNCTION_MUST_NOT_RESET')
        if initial_match_fn is not None:initial_match_fn(spec,root,initial_complete)
        while engine.step<500:
            if STOP_REQUESTED or(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-300:
                checkpoint=engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED',step=engine.step,cursor=engine.cursor,
                            checkpoint=checkpoint,actual=engine.counts,physical_requests=0,goal_met=False)
            try:row=engine.update()
            except TimeoutError:
                checkpoint=engine.checkpoint(f'LIMIT-step{engine.step:04d}')
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED',step=engine.step,cursor=engine.cursor,
                            checkpoint=checkpoint,actual=engine.counts,physical_requests=0,goal_met=False)
            if engine.step==1 or engine.step%10==0:
                write(root/'PROGRESS.private.json',dict(phase='OFFLINE_TRAINING',step=engine.step,cursor=engine.cursor,
                    parent_updates=1000,new_updates=engine.step,new_training_visits=engine.cursor,
                    actual=engine.counts,last=row,epoch=time.time(),physical_requests=0))
                print(json.dumps(dict(step=engine.step,cursor=engine.cursor,FM32=row['mean_FM32_loss'],
                                     physical_delta=row['physical_width_delta_loss'],wall_s=row['wall_s'])),flush=True)
            if engine.step in (250,500):checkpoints[engine.step]=engine.checkpoint(f'step{engine.step:04d}')
        engine.resume(checkpoints[500]);restored=verify_restored(engine,checkpoints[500])
        metrics=evaluate(engine,spec);complete=read(Path(checkpoints[500]['path'])/'COMPLETE.json')
        package=root/'RGB_PACKAGE';write_verified_package(package,checkpoints[500]['path'],complete=complete,
            active_template=engine.active,statistics=Path(spec['dataset'])/'STATISTICS.json',
            tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        return dict(schema='SLA_TIME_REPRESENTATION_PARENT1000_NEW500_ACTUAL_V1',exit_code=0,status='COMPLETE_NEW500',
            parent_updates=1000,parent_training_visits=32000,new_updates=engine.step,new_training_visits=engine.cursor,
            cumulative_candidate_updates=1500,cumulative_candidate_training_visits=48000,
            initial_checkpoint=initial,last_checkpoint=checkpoints[500],full_restore=restored,
            package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),
            time_codec_audit=spec['time_codec_audit'],prediction_file_sha256=sha(root/'TRAIN48_PREDICTIONS.private.json'),
            endpoint_RGB16_sha256=sha(root/'ENDPOINT_RGB16.private.json'),
            actual=engine.counts,diagnostic_backward_calls=1,diagnostic_backward_windows=engine.microbatch,
            metrics=metrics,gripper_lambda=spec['gripper_lambda'],inference_route=spec['inference_route'],
            representation=dict(codec=engine.bindings['codec_version'],recipe=engine.bindings['recipe'],
                                parameter_slot_shift=engine.bindings.get('parameter_slot_shift',0)),
            frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),
            wall_s=time.time()-started,CPU_s=time.process_time(),site=site,
            new_physical_requests=0,native_integrations=0,native_task_success=False,fresh_RGB_pending=True,goal_met=False)
    finally:engine.close()


def replay(spec,*,provider_class=None):
    from go2arx5_vla.sla_adapter.provider_time_v1 import Provider
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from openpi_client.image_tools import resize_with_pad
    if provider_class is not None:Provider=provider_class
    root=Path(spec['registered_root']);training=read(spec['training_result']['path'])
    assert sha(spec['training_result']['path'])==spec['training_result']['sha256']
    provider=Provider(training['package']['path'],manifest_sha256=training['package']['sha256'],
                      parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    dataset=MobileDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
                          max_cache_bytes=512*1024**2)
    records=[]
    endpoint=Path(spec['training_result']['path']).parent/'ENDPOINT_RGB16.private.json'
    assert sha(endpoint)==training['endpoint_RGB16_sha256']
    for row in read(endpoint):
        state,_,images,text=dataset.raw(row['index']);ep,_=dataset.rows[row['index']]
        result=provider.predict(state,images,text,dataset.frames[ep['id']]['T_policy_from_mount'],noise_id=row['noise'],previous_reference=row['previous_reference'])
        actual=dict(index=row['index'],noise=row['noise'],
            normalized_sha256=hashlib.sha256(result['normalized'].tobytes()).hexdigest(),
            physical_sha256=hashlib.sha256(result['physical_H40'].tobytes()).hexdigest())
        if any(actual[k]!=row[k]for k in actual):raise ValueError('FRESH_ORDINARY_RGB_RESTORE_DIFFERENCE')
        records.append(actual)
    return dict(exit_code=0,all_bit_exact=True,ordinary_RGB_generations=len(records),optimizer_loaded=False,
                optimizer_updates=0,optimization_visits=0,physical_requests=0,records=records,goal_met=False)


def main(*,train_fn=train,replay_fn=replay,RGB_operation='--time-rgb'):
    import signal
    def stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    s=read(sys.argv[2]);root=Path(s['registered_root']);lock=None
    try:
        lock=wait_resource_owner(s)
        write(root/'goal_state.json',dict(phase='ORDINARY_RGB_LOAD'if sys.argv[1]==RGB_operation else'OFFLINE_TRAINING_LOAD_PARENT1000',
            epoch=time.time(),source_sha=s['source_sha'],optimizer_updates=0,physical_requests=0))
        value=(replay_fn if sys.argv[1]==RGB_operation else train_fn)(s)
    except BaseException:
        # Partial effects remain authoritative in the per-update log/checkpoints.
        value=dict(exit_code=1,status='TECHNICAL_FAILURE_PRESERVED',failure=traceback.format_exc(),
                   latest_progress=read(root/'PROGRESS.private.json')if(root/'PROGRESS.private.json').exists()else None,
                   update_log=str(root/'work/updates.private.jsonl'),physical_requests=0,goal_met=False)
    finally:
        if lock:lock.close()
    write(root/'RESULT.private.json',value)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,value['exit_code']);raise SystemExit(value['exit_code'])


if __name__=='__main__':main()
