"""Fixed V5.2 RGB restoration and matched saved-input qualification.

Only B1/B2 make new CN38/ROLL96 predictions. Exact B0 outputs are reused;
scientific envelope failures are recorded and never remove an E2 control arm.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.close_goal_g1_v1 import wait_resource_owner,width_metrics
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
from go2arx5_vla.sla_adapter.mobile_successor_v52 import PLAN


def read(pin):
    if sha(pin['path'])!=pin['sha256']:raise ValueError('SEALED_V52_OFFLINE_PIN_CHANGED')
    return json.loads(Path(pin['path']).read_bytes())


def arrays(pin):
    if sha(pin['path'])!=pin['sha256']:raise ValueError('SEALED_V52_OFFLINE_ARRAY_CHANGED')
    with np.load(pin['path'],allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}


def run(spec):
    if (spec['plan_commit']!=PLAN or spec['scope']!='SLA_MOBILE_F2_V52'
            or spec['new_complete_generations_budget']!=600 or spec['physics']
            or spec['new_physical_requests_budget']or spec['new_optimizer_updates_budget']):
        raise PermissionError('FIXED_V52_600_OFFLINE_CALLS_ZERO_TRAIN_OR_NATIVE')
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.mobile_provider_pool_v52 import Pool
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.wm02.mobile_d0_v52 import boundary
    from go2arx5_vla.sla_adapter.shift7_cn38_r1_v4 import panel
    root=Path(spec['registered_root']);started=time.monotonic();site=validate(spec)
    from go2arx5_vla.sla_adapter.compilation_cache_binding_v1 import bind_declared_cache
    cache=bind_declared_cache(spec,jax.config)
    bank=read(spec['CN38_bank']);cases=panel(bank);entries={r['index']:r for r in bank['records']}
    roll=read(spec['ROLL96']);old_cn=read(spec['B0_R1_records'])
    old_by={(r['entry'],r['noise']):r for r in old_cn}
    if len(roll)!=96 or len(old_by)!=180:raise ValueError('EXACT_FIXED_ROLL96_CN180_REQUIRED')
    pool=Pool(spec);dataset=MobileDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=128*1024**2)
    counts={a:dict(attempted=0,completed=0,cap=16 if a=='B0'else 292)for a in('B0','B1','B2')}
    journal=Journal(root/'control/journal',root.name);records=[];restorations={}
    def counters():
        write(root/'GENERATION_COUNTS.private.json',dict(by_arm=counts,attempted=sum(v['attempted']for v in counts.values()),completed=sum(v['completed']for v in counts.values()),cap=600,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0))
    def generate(arm,key,state,images,language,frame,noise,previous):
        if(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-60:raise TimeoutError('FINITE_OFFLINE_V52_STOP_OR_DEADLINE')
        def effect():
            if counts[arm]['attempted']>=counts[arm]['cap']:raise PermissionError('FIXED_V52_PER_ARM_GENERATION_CAP')
            pool.select(arm);counts[arm]['attempted']+=1;counters();t=time.monotonic()
            value=pool.predict(state,images,language,frame,noise_id=noise,previous_reference=previous)
            if value['normalized'].shape!=(1,40,32)or value['physical_H40'].shape!=(40,10)or not np.isfinite(value['physical_H40']).all():raise ValueError('COMPLETE_FINITE_ORDINARY_H40_REQUIRED')
            path=root/'data'/(arm+'-'+key+'.npz')
            with path.open('xb')as f:np.savez(f,normalized=value['normalized'],physical_H40=value['physical_H40'])
            counts[arm]['completed']+=1;counters()
            return dict(path=str(path),sha256=sha(path),model_wall_s=time.monotonic()-t)
        return journal.once(arm+'-'+key,dict(package=spec['packages'][arm],noise=noise),effect)
    def record(arm,bank_name,key,output,state,policy,reference,expert=None,**context):
        value=arrays(output);action=value['physical_H40'];envelope=analyze(action,state,policy,reference)
        row=dict(arm=arm,bank=bank_name,key=key,output=output,envelope=envelope,endpoint=boundary(action),**context)
        if expert is not None:row['width']=width_metrics(action,expert)
        records.append(row)
        with(root/'RECORDS.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='FIXED_OFFLINE_QUALIFICATION',rows=len(records),by_arm=counts,new_updates=0,new_native_requests=0))
    # These 48 calls also qualify the shared-parent serving route against each
    # standalone endpoint. No additional warmup prediction is made.
    for arm in('B0','B1','B2'):
        training=read(spec['training_results'][arm]);endpoint=Path(spec['training_results'][arm]['path']).parent/'ENDPOINT_RGB16.private.json'
        if training['package']!=spec['packages'][arm]or sha(endpoint)!=training['endpoint_RGB16_sha256']:raise ValueError('EXACT_ORDINARY_RGB16_ENDPOINT_IDENTITY')
        old=json.loads(endpoint.read_bytes());assert len(old)==16
        for i,row in enumerate(old):
            state,_,images,text=dataset.raw(row['index']);ep,_=dataset.rows[row['index']]
            output=generate(arm,'RESTORE-%02d'%i,state,images,text,dataset.frames[ep['id']]['T_policy_from_mount'],row['noise'],row['previous_reference'])
            result=arrays(output)
            if (hashlib.sha256(result['normalized'].tobytes()).hexdigest()!=row['normalized_sha256']
                    or hashlib.sha256(result['physical_H40'].tobytes()).hexdigest()!=row['physical_sha256']):
                raise ValueError('SHARED_POOL_RGB16_DIFFERS_FROM_STANDALONE_ENDPOINT_'+arm)
        value=dict(arm=arm,package=spec['packages'][arm],all_bit_exact=True,generations=16,optimizer_loaded=False,new_updates=0,new_native_requests=0)
        path=root/(arm+'_RGB16_RESTORE.private.json');write(path,value);restorations[arm]=dict(path=str(path),sha256=sha(path))
    for case in cases:
        entry=entries[case['entry']];data=arrays(entry['input']);truth=arrays(entry['truth']);old=old_by[(case['entry'],case['noise'])]
        if old['input_sha256']!=entry['input']['sha256']or old['previous_reference']!=entry['previous_reference']:raise ValueError('B0_CN38_REUSE_INPUT_NOISE_REFERENCE_MISMATCH')
        for arm in('B0','B1','B2'):
            output=old['output']if arm=='B0'else generate(arm,'CN-%03d-%d'%(case['entry'],case['noise']),data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],entry['language'],data['T_policy_from_mount'],case['noise'],entry['previous_reference'])
            record(arm,'CN38',str((case['entry'],case['noise'])),output,data['state_eef10'],np.asarray(entry['policy_from_world']),truth['accepted_reference_TCP'],truth['physical_H40'],family=entry['family'],phase=entry['phase'],noise=case['noise'],input_sha256=entry['input']['sha256'],reused=arm=='B0')
    for i,entry in enumerate(roll):
        data=arrays(entry['input']);key='ROLL-%03d'%i
        for arm in('B0','B1','B2'):
            output=entry['B0_saved_output']if arm=='B0'else generate(arm,key,data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],entry['language'],data['T_policy_from_mount'],entry['noise_id'],entry['previous_reference'])
            record(arm,'ROLL96',key,output,data['state_eef10'],data['policy_from_world'],data['accepted_reference_TCP'],family=entry['family'],case=entry['case'],query=entry['query'],input_sha256=entry['input']['sha256'],reused=arm=='B0')
    if sum(v['completed']for v in counts.values())!=600 or len(records)!=828:raise ValueError('EXACT600_NEW_GENERATIONS_AND828_SAVED_CONDITIONS')
    summary={arm:{bank_name:dict(n=len(rows),full_H40_pass=sum(r['envelope']['original_full_H40_pass']for r in rows),failed_keys=[r['key']for r in rows if not r['envelope']['original_full_H40_pass']])for bank_name in('CN38','ROLL96')for rows in[[r for r in records if r['arm']==arm and r['bank']==bank_name]]}for arm in counts}
    result=dict(schema='SLA_MOBILE_V52_OFFLINE_ACTUAL',exit_code=0,source_sha=spec['source_sha'],plan_commit=PLAN,summary=summary,ordinary_RGB_restore=restorations,by_arm=counts,new_generations=600,parent_loads=1,optimizer_loaded=False,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,scientific_failures_do_not_filter_E2=True,wall_s=time.monotonic()-started,site=site,compilation_cache_binding=cache)
    write(root/'RESULT.private.json',result);return result


def main(spec):
    root=Path(spec['registered_root']);lock=None
    try:lock=wait_resource_owner(spec);result=run(spec)
    except BaseException:
        result=dict(exit_code=1,failure=traceback.format_exc(),counts=json.loads((root/'GENERATION_COUNTS.private.json').read_bytes())if(root/'GENERATION_COUNTS.private.json').exists()else {},new_updates=0,new_native_requests=0,new_integrations=0)
        write(root/'RESULT.private.json',result)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec,result['exit_code']);return result['exit_code']
