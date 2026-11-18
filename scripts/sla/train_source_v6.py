"""One source-native fixed1000 fit, followed by separate-process RGB restores."""
import hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
STOP=False


def read(path):return json.loads(Path(path).read_bytes())


def verify(engine,pin):
    import jax
    complete=read(Path(pin['path'])/'COMPLETE.json');leaves,_=jax.tree_util.tree_flatten_with_path((engine.active,engine.opt_state,engine.rng))
    if len(leaves)!=len(complete['arrays']):raise ValueError('COMPLETE_SOURCE_MODEL_ADAM_RNG_TREE_REQUIRED')
    for(key,value),row in zip(leaves,complete['arrays'],strict=True):
        path=Path(pin['path'])/row['file']
        if jax.tree_util.keystr(key)!=row['key']or sha(path)!=row['sha256']or not np.array_equal(np.asarray(value),np.load(path,allow_pickle=False)):
            raise ValueError('SOURCE_RESTORED_STATE_NOT_BIT_EXACT')
    return dict(bit_exact=True,arrays=len(leaves),step=engine.step,cursor=engine.cursor,optimizer_updates_added=0,model_Adam_RNG=True)


def source_train(spec):
    from go2arx5_vla.sla_adapter.source_training_v6 import SourceEngine
    from go2arx5_vla.sla_adapter.source_evaluation_v6 import evaluate
    root=Path(spec['registered_root']);start=time.monotonic();cpu=time.process_time();engine=SourceEngine(spec,root/'work')
    try:
        initial=engine.checkpoint('step0000');diagnostic=engine.diagnose();write(root/'GRADIENT_QUALIFICATION.private.json',diagnostic)
        initial_complete=read(Path(initial['path'])/'COMPLETE.json')
        active=[r for r in initial_complete['arrays']if r['key'].startswith('[0]')]
        if len(active)!=28 or engine.inventory['trainable_numel']!=52153376:raise ValueError('EXACT_SOURCE_TRAINABLE_SCOPE_REQUIRED')
        write(root/'INITIAL_SCOPE.private.json',dict(active=active,all_state_arrays=len(initial_complete['arrays']),
            trainable_numel=engine.inventory['trainable_numel'],fresh_Adam=True,initial_checkpoint=initial,bindings=engine.bindings))
        s0={split:evaluate(engine,root,split,endpoint='S0')for split in('TRAIN','SOURCE_DEV')}
        write(root/'S0_FROZEN.private.json',dict(initial_checkpoint=initial,metrics=s0,optimizer_updates=0,actual=engine.counts,
            RGB16=dict(path=str(root/'S0-RGB16.private.json'),sha256=sha(root/'S0-RGB16.private.json'))))
        while engine.step<1000:
            if STOP or(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-600:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_SOURCE_PREFIX_SAVED_NOT_RETRAINED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,new_native_requests=0)
            try:row=engine.update()
            except TimeoutError:
                saved=engine.checkpoint('LIMIT-step%04d'%engine.step)
                return dict(exit_code=75,status='BOUNDED_SOURCE_PREFIX_SAVED_NOT_RETRAINED',new_updates=engine.step,new_visits=engine.cursor,checkpoint=saved,actual=engine.counts,new_native_requests=0)
            if engine.step==1 or engine.step%10==0:write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='REAL_SOURCE_NATIVE_FM1000',step=engine.step,cursor=engine.cursor,actual=engine.counts,last=row))
            if engine.step in(250,500,1000):last=engine.checkpoint('step%04d'%engine.step)
            if engine.step in(250,500):
                engine.resume(last);write(root/('RECOVERY-step%04d.private.json'%engine.step),verify(engine,last))
        engine.resume(last);restored=verify(engine,last)
        s1000={split:evaluate(engine,root,split,endpoint='S1000')for split in('TRAIN','SOURCE_DEV')}
        complete=read(Path(last['path'])/'COMPLETE.json');shared=[]
        # Only explicitly shared LoRA factors may initialize M-SRC. Required
        # IO, source normalization and every Adam/RNG leaf stay source-only.
        for row in complete['arrays']:
            if row['key'].startswith('[0]')and 'PaliGemma'in row['key']and 'llm'in row['key']and('lora_a'in row['key']or'lora_b'in row['key']):
                shared.append(dict(row,path=str(Path(last['path'])/row['file'])))
        if not shared or any(any(x in r['key']for x in('action_in_proj','action_out_proj','time_mlp'))for r in shared):
            raise ValueError('SHARED_LORA_ONLY_TRANSFER_ALLOWLIST_REQUIRED')
        write(root/'SHARED_LORA.private.json',dict(source_checkpoint=last,arrays=shared,source_IO_norm_optimizer_excluded=True,
            target_IO_must_start_identically=True,initialization_only_not_inference_or_statistics_transfer=True))
        return dict(exit_code=0,status='SOURCE1000_SEALED_INDEPENDENT_RESTORES_AND_TEST_PENDING',source_sha=spec['source_sha'],plan_commit=spec['plan_commit'],
            new_updates=engine.step,new_visits=engine.cursor,initial_checkpoint=initial,last_checkpoint=last,full_restore=restored,
            S0=s0,S1000=s1000,actual=engine.counts,diagnostic=diagnostic,shared_lora=dict(path=str(root/'SHARED_LORA.private.json'),sha256=sha(root/'SHARED_LORA.private.json')),
            frozen_fingerprint_unchanged=engine.verify_frozen(),residency=engine.residency(),source_TEST_labels_read=False,
            RGB16={e:dict(path=str(root/(e+'-RGB16.private.json')),sha256=sha(root/(e+'-RGB16.private.json')))for e in('S0','S1000')},
            wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,new_native_requests=0,new_integrations=0)
    finally:engine.close()


def source_restore(spec):
    from go2arx5_vla.sla_adapter.source_training_v6 import SourceEngine
    root=Path(spec['registered_root']);training=read_pin(spec['training_spec']);parent=Path(training['registered_root']);result=read(parent/'RESULT.private.json')
    if result['exit_code']or result['new_updates']!=1000:raise ValueError('SEALED_FIXED1000_REQUIRED_BEFORE_INDEPENDENT_RESTORE')
    engine=SourceEngine(training,root/'work');endpoint=spec['endpoint'];pin=result['initial_checkpoint'if endpoint=='S0'else'last_checkpoint']
    try:
        engine.resume(pin);full=verify(engine,pin);rows=read_pin(result['RGB16'][endpoint]);comparisons=[]
        for row in rows:
            normalized,_=engine.generate(row['index'],row['noise_id']);reference=np.load(row['array']['path'],allow_pickle=False)['normalized']
            difference=float(np.max(abs(normalized-reference)));exact=bool(np.array_equal(normalized,reference))
            comparisons.append(dict(window=row['window'],bit_exact=exact,max_abs=difference))
            write(root/'PROGRESS.private.json',dict(endpoint=endpoint,completed=len(comparisons),actual=engine.counts))
        passed=all(r['bit_exact']for r in comparisons)
        return dict(exit_code=0,status='INDEPENDENT_SOURCE_RGB_RESTORE_COMPLETE',endpoint=endpoint,passed=passed,
            comparisons=comparisons,full_restore=full,actual=engine.counts,optimizer_updates_added=0,new_native_requests=0)
    finally:engine.close()


def main(spec,*,restore=False):
    from scripts.sla.close_goal_g1_v1 import wait_resource_owner
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    root=Path(spec['registered_root']);lock=None
    def stop(*_):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        lock=wait_resource_owner(spec);site=validate(spec)
        result=source_restore(spec)if restore else source_train(spec);result['site']=site
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_SOURCE_TECHNICAL_FAILURE_NO_AUTORETRY',failure=traceback.format_exc(),
            latest_progress=read(root/'PROGRESS.private.json')if(root/'PROGRESS.private.json').exists()else None,
            actual_update_log=str(root/'work/updates.private.jsonl'),new_native_requests=0,new_integrations=0)
    finally:
        if lock:lock.close()
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
