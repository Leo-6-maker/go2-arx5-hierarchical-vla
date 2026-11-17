"""Fixed377 target conditions; reuse372 frozen B1 outputs, add only21 calls."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.close_goal_g1_v1 import wait_resource_owner,width_metrics
from scripts.sla.mobile_offline_v52 import read,arrays
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.target_contract_v6 import PLAN,B1
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze


def run(spec):
    if(spec['plan_commit']!=PLAN or spec['scope']!='SLA_V6_TARGET_NATIVE' or spec['new_complete_generations_budget']!=615
       or spec['physics']or spec['new_physical_requests_budget']or spec['new_optimizer_updates_budget']):
        raise PermissionError('FIXED615_NEW_TARGET_CALLS_ZERO_OPTIMIZER_OR_NATIVE')
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.target_pool_v6 import Pool
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.wm02.mobile_d0_v52 import boundary
    from go2arx5_vla.sla_adapter.shift7_cn38_r1_v4 import panel
    root=Path(spec['registered_root']);start=time.monotonic();site=validate(spec)
    jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
    bank=read(spec['CN38_bank']);entries={r['index']:r for r in bank['records']};cases=panel(bank);roll=read(spec['ROLL96']);rejects=read(spec['E2_REJECT5'])
    saved_pin=spec['B1_saved_offline']
    if sha(saved_pin['path'])!=saved_pin['sha256']:raise ValueError('SEALED_OLD_B1_OFFLINE_JSONL_REQUIRED')
    saved=[r for line in Path(saved_pin['path']).read_text().splitlines()for r in[json.loads(line)]if r['arm']=='B1'];old={(r['bank'],r['key']):r for r in saved}
    if len(cases)!=180 or len(roll)!=96 or len(rejects)!=5 or len(old)!=276:raise ValueError('EXACT_CN180_ROLL96_REJECT5_AND_FROZEN_B1_REUSE_REQUIRED')
    if spec['packages']['T-B1']['sha256']!=B1:raise ValueError('EXACT_FROZEN_B1_CONTROL_REQUIRED')
    pool=Pool(spec);dataset=MobileDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=128*1024**2)
    counts={arm:dict(attempted=0,completed=0,cap=21 if arm=='T-B1'else 297)for arm in('T-B1','T-FM','T-H40')};records=[];restores={};journal=Journal(root/'control/journal',root.name)
    def counters():write(root/'GENERATION_COUNTS.private.json',dict(by_arm=counts,attempted=sum(r['attempted']for r in counts.values()),completed=sum(r['completed']for r in counts.values()),cap=615,new_updates=0,new_native_requests=0))
    def generate(arm,key,state,images,text,frame,noise,previous):
        if(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-120:raise TimeoutError('FINITE_TARGET_OFFLINE_STOP_OR_DEADLINE')
        def effect():
            if counts[arm]['attempted']>=counts[arm]['cap']:raise PermissionError('FIXED_TARGET_PER_ARM_CAP')
            pool.select(arm);counts[arm]['attempted']+=1;counters();started=time.monotonic()
            result=pool.predict(state,images,text,frame,noise_id=noise,previous_reference=previous)
            if result['normalized'].shape!=(1,40,32)or result['physical_H40'].shape!=(40,10)or not np.isfinite(result['physical_H40']).all():raise ValueError('FULL_FINITE_TARGET_H40_REQUIRED')
            path=root/'data'/(arm+'-'+key+'.npz')
            with path.open('xb')as f:np.savez(f,normalized=result['normalized'],physical_H40=result['physical_H40'])
            counts[arm]['completed']+=1;counters();return dict(path=str(path),sha256=sha(path),model_wall_s=time.monotonic()-started)
        return journal.once(arm+'-'+key,dict(package=spec['packages'][arm],noise=noise),effect)
    def record(arm,name,key,output,state,policy,reference,expert=None,**context):
        action=arrays(output)['physical_H40'];envelope=analyze(action,state,policy,reference)
        row=dict(arm=arm,bank=name,key=key,output=output,envelope=envelope,endpoint=boundary(action),**context)
        if expert is not None:row['width']=width_metrics(action,expert)
        records.append(row)
        with(root/'RECORDS.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='FIXED_TARGET_OFFLINE377',rows=len(records),by_arm=counts,new_updates=0,new_native_requests=0))
    for arm in counts:
        training=read(spec['training_results'][arm]);endpoint=Path(spec['training_results'][arm]['path']).parent/'ENDPOINT_RGB16.private.json'
        if training['package']!=spec['packages'][arm]or sha(endpoint)!=training['endpoint_RGB16_sha256']:raise ValueError('ACTUAL_TARGET_ENDPOINT_RGB16_IDENTITY_REQUIRED')
        rows=json.loads(endpoint.read_bytes())
        if len(rows)!=16:raise ValueError('EXACT16_INDEPENDENT_ORDINARY_RGB_RESTORES_REQUIRED')
        for i,row in enumerate(rows):
            state,_,images,text=dataset.raw(row['index']);ep,_=dataset.rows[row['index']]
            output=generate(arm,'RESTORE-%02d'%i,state,images,text,dataset.frames[ep['id']]['T_policy_from_mount'],row['noise'],row['previous_reference']);result=arrays(output)
            if hashlib.sha256(result['normalized'].tobytes()).hexdigest()!=row['normalized_sha256']or hashlib.sha256(result['physical_H40'].tobytes()).hexdigest()!=row['physical_sha256']:
                raise ValueError('TARGET_POOL_RGB16_DIFFERS_FROM_STANDALONE_'+arm)
        path=root/(arm+'-RGB16_RESTORE.private.json');write(path,dict(arm=arm,package=spec['packages'][arm],all_bit_exact=True,generations=16,optimizer_loaded=False,new_updates=0,new_native_requests=0));restores[arm]=dict(path=str(path),sha256=sha(path))
    for case in cases:
        entry=entries[case['entry']];data=arrays(entry['input']);truth=arrays(entry['truth']);key=str((case['entry'],case['noise']))
        prior=old['CN38',key]
        if prior['input_sha256']!=entry['input']['sha256']or prior['noise']!=case['noise']:raise ValueError('FROZEN_B1_CN_INPUT_NOISE_REUSE_IDENTITY')
        for arm in counts:
            output=prior['output']if arm=='T-B1'else generate(arm,'CN-%03d-%d'%(case['entry'],case['noise']),data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],entry['language'],data['T_policy_from_mount'],case['noise'],entry['previous_reference'])
            record(arm,'CN38',key,output,data['state_eef10'],np.asarray(entry['policy_from_world']),truth['accepted_reference_TCP'],truth['physical_H40'],family=entry['family'],phase=entry['phase'],noise=case['noise'],input_sha256=entry['input']['sha256'],previous_reference=entry['previous_reference'],reused=arm=='T-B1')
    for i,entry in enumerate(roll):
        data=arrays(entry['input']);key='ROLL-%03d'%i;prior=old['ROLL96',key]
        if prior['input_sha256']!=entry['input']['sha256']or prior['case']!=entry['case']or prior['query']!=entry['query']:raise ValueError('FROZEN_B1_ROLL_REUSE_IDENTITY')
        for arm in counts:
            output=prior['output']if arm=='T-B1'else generate(arm,key,data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],entry['language'],data['T_policy_from_mount'],entry['noise_id'],entry['previous_reference'])
            record(arm,'ROLL96',key,output,data['state_eef10'],data['policy_from_world'],data['accepted_reference_TCP'],family=entry['family'],case=entry['case'],query=entry['query'],input_sha256=entry['input']['sha256'],previous_reference=entry['previous_reference'],reused=arm=='T-B1')
    for i,entry in enumerate(rejects):
        request=read(entry['request']);data=arrays(request['input']);meta=read(entry['metadata']);native=read(entry['query_actual'])
        if request['input']!=meta['input']or request['previous_reference']!=meta['previous_reference']or native['full_H40_compiled']is not False:
            raise ValueError('REAL_SAVED_REJECT_INPUT_CLOCK_AND_FIRST_FAILURE_REQUIRED')
        for arm in counts:
            output=generate(arm,'E2-REJECT-%02d'%i,data['state_eef10'],[data['body_rgb'],data['wrist_rgb']],request['language'],data['T_policy_from_mount'],request['noise_id'],request['previous_reference'])
            record(arm,'E2_REJECT5',entry['id'],output,data['state_eef10'],data['policy_from_world'],data['accepted_reference_TCP'],input_sha256=request['input']['sha256'],previous_reference=request['previous_reference'],reused=False,original_rejection=entry['query_actual'])
    if sum(c['completed']for c in counts.values())!=615 or len(records)!=843:raise ValueError('EXACT615_NEW_GENERATIONS_AND843_SAVED_CONDITIONS_REQUIRED')
    result=dict(exit_code=0,status='FIXED_TARGET_OFFLINE_SEALED_SCIENCE_FAILURES_RETAINED',source_sha=spec['source_sha'],plan_commit=PLAN,
        ordinary_RGB_restore=restores,by_arm=counts,new_generations=615,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,
        source_records=843,reused_B1_predictions=276,TRAIN96_reused_for_all_arms=True,scientific_failures_do_not_filter_screen=True,wall_s=time.monotonic()-start,site=site)
    write(root/'RESULT.private.json',result);return result


def main(spec):
    root=Path(spec['registered_root']);lock=None
    try:lock=wait_resource_owner(spec);result=run(spec)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_TARGET_OFFLINE_TECHNICAL_FAILURE_NO_AUTORETRY',failure=traceback.format_exc(),
            counts=json.loads((root/'GENERATION_COUNTS.private.json').read_bytes())if(root/'GENERATION_COUNTS.private.json').exists()else {},new_updates=0,new_native_requests=0)
        write(root/'RESULT.private.json',result)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec,result['exit_code']);return result['exit_code']
