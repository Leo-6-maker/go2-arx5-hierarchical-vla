"""Fixed500 V6 target arm, no additional optimizer smoke or scientific retry."""
import hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
STOP=False


def read(path):return json.loads(Path(path).read_bytes())


def train(spec):
    import jax
    from go2arx5_vla.sla_adapter.target_training_v6 import TargetEngine,B1
    from go2arx5_vla.sla_adapter.provider_target_v6 import export
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from scripts.sla.train_pd4_matched_v71 import verify_restored
    from scripts.sla.train_close_time_v1 import evaluate
    root=Path(spec['registered_root']);site=validate(spec);started=time.monotonic();cpu=time.process_time()
    diagnostic=read(spec['real_gradient_qualification']['path'])
    if(sha(spec['real_gradient_qualification']['path'])!=spec['real_gradient_qualification']['sha256']or diagnostic['exit_code']
       or not diagnostic['backward_complete']or diagnostic['new_updates']or diagnostic['diagnostic_windows']!=4):
        raise PermissionError('SEALED_REAL_FULL_H40_ZERO_UPDATE_GRADIENT_REQUIRED')
    engine=TargetEngine(spec,root/'work')
    try:
        initial=engine.checkpoint('step0000');complete=read(Path(initial['path'])/'COMPLETE.json');parent=read(Path(spec['B1_parent_package']['path'])/'MANIFEST.private.json')
        active=[r for r in complete['arrays']if r['key'].startswith('[0]')]
        if len(complete['arrays'])!=87 or len(active)!=28 or any(any(a[k]!=b[k]for k in('key','shape','dtype','sha256'))for a,b in zip(active,parent['adapter_arrays'],strict=True)):
            raise ValueError('REAL_B1_PLUS_FRESH_MATCHED_ADAM_REQUIRED')
        noise,tau=engine.draw_bank(engine.rng,np.arange(32,dtype=np.uint32));first=dict(initial_checkpoint=initial,all87_array_hashes=[r['sha256']for r in complete['arrays']],
            active28=active,visit_map_sha256=engine.bindings['visit_map_sha256'],LR_sha256=engine.bindings['lr_sha256'],noise_sha256=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest(),
            time_sha256=hashlib.sha256(np.asarray(tau).tobytes()).hexdigest(),first32_indices=engine.visit_map[:32].tolist(),parent_package_sha256=B1,optimizer_updates=0,diagnostic_backward_calls_added=0)
        if spec.get('matched_control_root'):
            control=Path(spec['matched_control_root']);control_result=read(control/'RESULT.private.json')
            if control_result['exit_code']or control_result['new_updates']!=500:raise ValueError('ACTUAL_COMPLETED_CONTROL_REQUIRED_FOR_MATCHED_ARM')
            other=read(control/'INITIAL_MATCH.private.json')
            if any(first[k]!=other[k]for k in('all87_array_hashes','visit_map_sha256','LR_sha256','noise_sha256','time_sha256','first32_indices','parent_package_sha256')):
                raise ValueError('ALL87_AND_ABSOLUTE_CURSORS_RANDOM_BANKS_MUST_MATCH')
            first['matched_control_result_sha256']=sha(control/'RESULT.private.json')
        write(root/'INITIAL_MATCH.private.json',first)
        while engine.step<500:
            if STOP or(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-300:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED_NOT_RETRAINED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,new_native_requests=0)
            try:row=engine.update()
            except TimeoutError:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_PREFIX_SAVED_NOT_RETRAINED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,new_native_requests=0)
            if engine.step==1 or engine.step%10==0:
                write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='REAL_V6_TARGET_TRAINING',arm=spec['arm'],step=engine.step,cursor=engine.cursor,actual=engine.counts,last=row))
            if engine.step in(250,500):last=engine.checkpoint('step%04d'%engine.step)
        engine.resume(last);restore=verify_restored(engine,last);metrics=evaluate(engine,spec)
        complete=read(Path(last['path'])/'COMPLETE.json');package=root/'RGB_PACKAGE';export(package,last['path'],complete=complete,active_template=engine.active,statistics=Path(spec['dataset'])/'STATISTICS.json',tokenizer=spec['tokenizer']['path'],parent_manifest=spec['parent_manifest'])
        return dict(schema='SLA_V6_REAL_TARGET500_ACTUAL',exit_code=0,status='FIXED500_COMPLETE_OFFLINE_NATIVE_PENDING',arm=spec['arm'],source_sha=spec['source_sha'],plan_commit=spec['plan_commit'],
            parent_package=spec['B1_parent_package'],new_updates=engine.step,new_visits=engine.cursor,parent_cumulative_updates=2000,cumulative_updates=2500,cumulative_visits=80000,
            initial_checkpoint=initial,last_checkpoint=last,full_restore=restore,package=dict(path=str(package),sha256=sha(package/'MANIFEST.private.json')),actual=engine.counts,
            TRAIN96_metrics=metrics,TRAIN96_predictions=dict(path=str(root/'TRAIN48_PREDICTIONS.private.json'),sha256=sha(root/'TRAIN48_PREDICTIONS.private.json')),
            endpoint_RGB16_sha256=sha(root/'ENDPOINT_RGB16.private.json'),frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),
            wall_s=time.monotonic()-started,CPU_s=time.process_time()-cpu,site=site,ordinary_RGB_restore_pending=True,new_native_requests=0,new_integrations=0)
    finally:engine.close()


def main(spec):
    from scripts.sla.close_goal_g1_v1 import wait_resource_owner
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    root=Path(spec['registered_root']);lock=None
    def stop(*_):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        lock=wait_resource_owner(spec);result=train(spec)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_TARGET_TECHNICAL_FAILURE_NO_AUTORETRY',failure=traceback.format_exc(),
            latest_progress=read(root/'PROGRESS.private.json')if(root/'PROGRESS.private.json').exists()else None,
            actual_update_log=str(root/'work/updates.private.jsonl'),new_native_requests=0,new_integrations=0)
    finally:
        if lock:lock.close()
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
