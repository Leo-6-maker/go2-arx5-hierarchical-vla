"""Bounded V5.2 500-update endpoint, retaining actual model/optimizer effects."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.close_goal_g1_v1 import wait_resource_owner
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
STOP_REQUESTED=False


def read(p):return json.loads(Path(p).read_bytes())


def train(spec):
    import jax
    import optax
    from go2arx5_vla.sla_adapter.mobile_successor_v52 import SuccessorEngine
    from go2arx5_vla.sla_adapter.provider_mobile_v52 import export
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.batch_v2 import host_batch,device_batch
    from scripts.sla.train_close_time_v1 import evaluate
    from scripts.sla.train_pd4_matched_v71 import verify_restored
    root=Path(spec['registered_root']);started=time.monotonic();cpu=time.process_time();site=validate(spec)
    # A sealed population must predate even the new optimizer's construction.
    split=spec['evaluation_split'];population=read(split['path'])
    if sha(split['path'])!=split['sha256']or not population['frozen_before_training']or population['DEV_count']!=32 or population['TEST_count']!=40:
        raise ValueError('ACTUAL_SEALED_LAYOUT_SPLIT_REQUIRED_BEFORE_TRAINING')
    engine=SuccessorEngine(spec,root/'work')
    try:
        initial=engine.checkpoint('step0000');manifest=read(Path(initial['path'])/'COMPLETE.json');parent=read(Path(spec['SHIFT7_parent_package']['path'])/'MANIFEST.private.json')
        active=[r for r in manifest['arrays']if r['key'].startswith('[0]')]
        if len(active)!=28 or any(any(a[k]!=b[k]for k in('key','shape','dtype','sha256'))for a,b in zip(active,parent['adapter_arrays'],strict=True)):
            raise ValueError('STEP_ZERO_MUST_BE_REAL_SHIFT7_NOT_C20_PARENT')
        host=host_batch(engine.dataset,engine.tokenizer,engine.visit_map,0,4);obs,actions=device_batch(host);noise,tau=engine.draw_bank(engine.rng,np.arange(32,dtype=np.uint32))
        (loss,aux),gradient=engine.grad(engine.active,engine.frozen,obs,actions,noise[:4],tau[:4]);norm=float(optax.global_norm(gradient))
        if not np.isfinite(np.asarray([loss,*aux,norm])).all()or norm<=0:raise ValueError('REAL_V52_DIAGNOSTIC_GRADIENT')
        write(root/'INITIAL_MATCH.private.json',dict(parent_package=spec['SHIFT7_parent_package'],initial_checkpoint=initial,active_arrays=active,all_state_array_hashes=[r['sha256']for r in manifest['arrays']],visit_map_sha256=engine.bindings['visit_map_sha256'],LR_sha256=engine.bindings['lr_sha256'],noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),tau_sha256=hashlib.sha256(np.asarray(tau).tobytes()).hexdigest(),state_dropout=host['state_token_dropped'],first_indices=host['indices'],diagnostic_backward_calls=1,diagnostic_backward_visits=4,diagnostic_loss=float(loss),diagnostic_edge=float(aux[3]),gradient_norm=norm,optimization_updates=0,optimization_visits=0,fresh_Adam=True))
        del gradient
        while engine.step<500:
            if STOP_REQUESTED or(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-300:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,diagnostic_backward_calls=1,diagnostic_backward_visits=4,new_native_requests=0)
            try:engine.update()
            except TimeoutError:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,diagnostic_backward_calls=1,diagnostic_backward_visits=4,new_native_requests=0)
            if engine.step==1 or engine.step%10==0:
                write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='V52_REAL_TRAINING',step=engine.step,cursor=engine.cursor,arm=spec['arm'],parent_cumulative_updates=1500,new_updates=engine.step,new_visits=engine.cursor,actual=engine.counts))
            if engine.step in(250,500):last=engine.checkpoint('step%04d'%engine.step)
        engine.resume(last);restore=verify_restored(engine,last)
        metrics=evaluate(engine,spec)
        complete=read(Path(last['path'])/'COMPLETE.json');package=root/'RGB_PACKAGE';export(package,last['path'],complete=complete,active_template=engine.active,statistics=Path(spec['dataset'])/'STATISTICS.json',tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        return dict(schema='SLA_MOBILE_V52_REAL500_ACTUAL',exit_code=0,status='FIXED500_COMPLETE',arm=spec['arm'],source_sha=spec['source_sha'],plan_commit=spec['plan_commit'],parent_package=spec['SHIFT7_parent_package'],parent_cumulative_updates=1500,parent_cumulative_visits=48000,new_updates=engine.step,new_visits=engine.cursor,cumulative_updates=2000,cumulative_visits=64000,endpoint_lambda=spec['endpoint_lambda'],initial_checkpoint=initial,last_checkpoint=last,full_restore=restore,package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),actual=engine.counts,diagnostic_backward_calls=1,diagnostic_backward_visits=4,TRAIN96_metrics=metrics,TRAIN96_predictions=dict(path=str(root/'TRAIN48_PREDICTIONS.private.json'),sha256=sha(root/'TRAIN48_PREDICTIONS.private.json')),endpoint_RGB16_sha256=sha(root/'ENDPOINT_RGB16.private.json'),frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu,site=site,ordinary_RGB_restore_pending=True,new_native_requests=0,new_integrations=0)
    finally:engine.close()


def replay(spec):
    from scripts.sla.train_close_time_v1 import replay as original
    from go2arx5_vla.sla_adapter.provider_mobile_v52 import Provider
    if spec['arm']=='B0':
        from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import Provider as FrozenProvider
        return original(spec,provider_class=FrozenProvider)
    return original(spec,provider_class=Provider)


def main(operation,spec):
    import signal
    def stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    root=Path(spec['registered_root']);lock=None
    try:
        lock=wait_resource_owner(spec)
        write(root/'goal_state.json',dict(phase='V52_LOAD_REAL_SHIFT7_1500_OR_SUCCESSOR_RGB',epoch=time.time(),source_sha=spec['source_sha']))
        result=(train if operation=='--train'else replay)(spec)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_TECHNICAL_FAILURE',failure=traceback.format_exc(),latest_progress=read(root/'PROGRESS.private.json')if(root/'PROGRESS.private.json').exists()else None,update_log=str(root/'work/updates.private.jsonl'),new_native_requests=0,new_integrations=0)
    finally:
        if lock:lock.close()
    write(root/'RESULT.private.json',result)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec,result['exit_code']);return result['exit_code']
