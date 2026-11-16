"""One finite shared-parent dual-adapter inference identity; no optimizer/native API."""
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.close_goal_g1_v1 import wait_resource_owner,width_metrics
from scripts.sla.export_mfree_c20_v1 import _reference
from scripts.sla.hybrid_saved96_v1 import summarize
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.hybrid_eef_grip_v1 import compose,coordination
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
read=lambda p:json.loads(Path(p).read_bytes())


def load_pair(spec):
    from go2arx5_vla.sla_adapter.provider_close_v1 import Provider as EEF
    from go2arx5_vla.sla_adapter.provider_time_v1 import Provider as Time
    from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import Provider as Shift
    from go2arx5_vla.sla_adapter.resident_close_pair_v1 import attach
    a=spec['FM_package'];b=spec['grip_package']
    one=EEF(a['path'],manifest_sha256=a['sha256'],parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    if one.manifest['bindings']['gripper_lambda']!=0:raise ValueError('PURE_FM_EEF_REQUIRED')
    two=attach(one,b,Time if spec['selected_combo']=='FM_EEF_TIME_GRIP'else Shift)
    return one,two


def run(spec):
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    root=Path(spec['registered_root']);start=time.time();cpu=time.process_time();site=validate(spec)
    bank=spec['CN38_bank']
    if sha(bank['path'])!=bank['sha256']:raise ValueError('FROZEN_CN38_BANK_REQUIRED')
    inputs=read(bank['path']);one,two=load_pair(spec);rows=[];attempts=0;completed=0;replays=[]
    write(root/'MODEL_LOAD_ACTUAL.private.json',dict(parent_loads=1,adapter_packages=2,optimizer_loaded=False,
        selected_combo=spec['selected_combo'],GPU_memory=jax.devices()[0].memory_stats()))
    def generate(state,images,language,frame,noise,reference):
        nonlocal attempts,completed
        outputs=[];latencies=[]
        for provider,extra in ((one,{}),(two,dict(previous_reference=reference))):
            if(root/'STOP.json').exists()or time.time()>=spec['deadline_epoch']-60:raise TimeoutError('FINITE_DUAL_ADAPTER_STOP_OR_DEADLINE')
            if attempts>=376:raise PermissionError('FINITE360_PLUS16_MODEL_GENERATION_CAP')
            attempts+=1;t=time.monotonic();value=provider.predict(state,images,language,frame,noise_id=noise,**extra)
            latencies.append(time.monotonic()-t);completed+=1;outputs.append(value)
            write(root/'GENERATION_COUNTS.private.json',dict(attempted=attempts,completed=completed,
                generation_cap=376,new_updates=0,new_native_requests=0))
        return outputs,latencies
    for entry in inputs['records']:
        if sha(entry['input']['path'])!=entry['input']['sha256']or sha(entry['truth']['path'])!=entry['truth']['sha256']:
            raise ValueError('FROZEN_CURRENT_INPUT_AND_DIAGNOSTIC_TRUTH')
        with np.load(entry['input']['path'],allow_pickle=False)as z:
            state=z['state_eef10'].copy();images=[z['body_rgb'].copy(),z['wrist_rgb'].copy()];frame=z['T_policy_from_mount'].copy()
        with np.load(entry['truth']['path'],allow_pickle=False)as z:expert=z['physical_H40'].copy();reference=z['accepted_reference_TCP'].copy()
        for noise in (0,1):
            outputs,latencies=generate(state,images,entry['language'],frame,noise,entry['previous_reference'])
            action=compose(outputs[0]['physical_H40'],outputs[1]['physical_H40']);env=analyze(action,state,np.asarray(entry['policy_from_world']),reference)
            f=root/'data'/f"{entry['index']:03d}-{noise}.npz"
            with f.open('xb')as handle:np.savez(handle,FM_physical_H40=outputs[0]['physical_H40'],grip_physical_H40=outputs[1]['physical_H40'],physical_H40=action)
            rows.append(dict(index=entry['index'],episode=entry['episode'],family=entry['family'],phase=entry['phase'],anchor=entry['anchor'],noise=noise,
                input_sha256=entry['input']['sha256'],previous_reference=entry['previous_reference'],
                envelope=env,width=width_metrics(action,expert),coordination=coordination(action,expert,state,entry['previous_reference'],entry['policy_from_world']),
                EEF_maximum_position_error_m=float(np.linalg.norm(action[:,:3]-expert[:,:3],axis=1).max()),
                output=dict(path=str(f),sha256=sha(f)),branch_wall_s=latencies,EEF_source='FM500',grip_source=spec['selected_combo']))
            write(root/'PROGRESS.private.json',dict(completed_conditions=len(rows),attempted_generations=attempts,completed_generations=completed,
                generation_cap=376,new_updates=0,new_native_requests=0,epoch=time.time()))
    if len(rows)!=180 or completed!=360:raise ValueError('EXACT180_CONDITIONS_360_BRANCH_GENERATIONS')
    write(root/'CN38_RECORDS.private.json',rows)
    # Separately declared16 calls: four fixed oldRGB anchors, two noises, two branches.
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from openpi_client.image_tools import resize_with_pad
    d=MobileDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,max_cache_bytes=128*1024**2)
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    a=read_pin(spec['FM_RGB16']);b=read_pin(spec['grip_RGB16'])
    for olda,oldb in zip(a[:8],b[:8]):
        if(olda['index'],olda['noise'])!=(oldb['index'],oldb['noise']):raise ValueError('FIXED_FRESH_REPLAY_IDENTITY')
        state,_,images,text=d.raw(olda['index']);ep,_=d.rows[olda['index']]
        values,_=generate(state,images,text,d.frames[ep['id']]['T_policy_from_mount'],olda['noise'],oldb['previous_reference'])
        for value,old in zip(values,(olda,oldb)):
            with np.load(old['output']['path'],allow_pickle=False)as z:
                if not np.array_equal(value['physical_H40'],z['physical_H40'])or not np.array_equal(value['normalized'],z['normalized']):
                    raise ValueError('SHARED_RESIDENT_ADAPTER_DIFFERS_FROM_SEALED_STANDALONE_OUTPUT')
        replays.append(dict(index=olda['index'],noise=olda['noise'],both_branch_outputs_bit_exact=True))
    if completed!=376 or len(replays)!=8:raise ValueError('FINITE360_PLUS16_GENERATION_ACCOUNTING')
    result=dict(schema='MFREE_HYBRID_CN38_EPISODE_DIAGNOSTIC_ACTUAL_V1',exit_code=0,selected_combo=spec['selected_combo'],
        summary=summarize(rows),by_family={f:summarize([r for r in rows if r['family']==f])for f in ('F1','F2')},
        by_episode={e:dict(family=next(r['family']for r in rows if r['episode']==e),**summarize([r for r in rows if r['episode']==e]))for e in sorted({r['episode']for r in rows})},
        split='EPISODE_HELDOUT_DIAGNOSTIC_NOT_BLIND_TEST',episodes=18,conditions=180,parent_loads=1,adapter_packages=2,
        attempted_generations=attempts,completed_generations=completed,CN38_branch_generations=360,shared_replay_generations=16,
        shared_resident_replay=replays,peak_GPU_memory=jax.devices()[0].memory_stats(),site=site,
        world_or_native_seam_qualified=False,STOP_TTL_partial_P10_native_lifecycle_qualified=False,
        optimizer_loaded=False,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,
        wall_s=time.time()-start,CPU_s=time.process_time()-cpu,physical_admission=False,goal_met=False)
    write(root/'RESULT.private.json',result);return result


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root']);lock=None
    try:
        lock=wait_resource_owner(s);result=run(s)
    except BaseException:
        counts=read(root/'GENERATION_COUNTS.private.json')if(root/'GENERATION_COUNTS.private.json').exists()else{}
        result=dict(exit_code=1,status='PRESERVED_TECHNICAL_FAILURE_NO_AUTO_REPLAY',failure=traceback.format_exc(),
            actual_counts=counts,new_updates=0,new_native_requests=0,new_integrations=0,goal_met=False)
        write(root/'RESULT.private.json',result)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,result['exit_code']);raise SystemExit(result['exit_code'])
