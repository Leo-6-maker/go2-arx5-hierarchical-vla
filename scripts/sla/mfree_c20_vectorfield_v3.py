"""Fixed-field numerical audit: actual error32, official full and cached suffix.

No optimizer or gradient is constructed. No sample_actions call or physics is
performed. Official source slices are extracted with exact SHA guards, then
only their final vector field is returned at a supplied identical x_tau/tau.
"""
import ast,contextlib,hashlib,inspect,json,os,sys,textwrap,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import OwnerLock,write
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha
PI0_SHA='24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342'

def extracted_fields(batch_module,pi0_module):
    """Keep the actual original network expressions; inject no new math."""
    training=inspect.getsource(batch_module.error32)
    network=training[training.index('    prefix, prefix_mask, prefix_ar ='):training.index('    return jnp.square')]
    train='def train_field(model, observation, x_t, flow_time):\n    observation = upstream.preprocess_observation(None, observation, train=False)\n'+network+'    return predicted\n'
    official=textwrap.dedent(inspect.getsource(pi0_module.Pi0.compute_loss));part=official[official.index('    prefix_tokens, prefix_mask, prefix_ar_mask ='):official.index('    return jnp.mean')]
    full='def official_field(self, observation, x_t, time):\n    observation = _model.preprocess_observation(None, observation, train=False)\n'+part+'    return v_t\n'
    sample=textwrap.dedent(inspect.getsource(pi0_module.Pi0.sample_actions));prefix=sample[sample.index('    prefix_tokens, prefix_mask, prefix_ar_mask ='):sample.index('    def step')]
    fill='def prefix_field(self, observation):\n    observation = _model.preprocess_observation(None, observation, train=False)\n'+prefix+'    return observation, prefix_tokens, prefix_mask, kv_cache\n'
    parsed=ast.parse(sample);step=next(n for n in parsed.body[0].body if isinstance(n,ast.FunctionDef)and n.name=='step');body=step.body[1:-1] # omit carry unpacking and Euler integration
    # The original nested function expects scalar time and broadcasts it. All
    # audit inputs use one constant tau per batch, exactly as sample_actions.
    cached='def cache_field(self, observation, prefix_tokens, prefix_mask, kv_cache, x_t, time):\n    batch_size = observation.state.shape[0]\n'+textwrap.indent('\n'.join(ast.unparse(n)for n in body),'    ')+'\n    return v_t\n'
    source=train+'\n'+full+'\n'+fill+'\n'+cached
    globals_=dict(pi0_module.__dict__);globals_.update(upstream=pi0_module._model)
    exec(compile(source,'<PINNED_C20_VECTOR_FIELDS>','exec'),globals_)
    return {k:globals_[k]for k in ('train_field','official_field','prefix_field','cache_field')},source

def difference(a,b):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return dict(max_abs=float(abs(d).max()),active7_max_abs=float(abs(d[...,:7]).max()),padding25_max_abs=float(abs(d[...,7:]).max()),RMS=float(np.sqrt(np.mean(d*d))),bit_exact=bool(np.array_equal(a,b)))

def run(s):
    import jax,jax.numpy as jnp,flax.nnx as nnx
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter import batch_v2
    from go2arx5_vla.sla_adapter.mfree_training_v1 import MobileDataset
    from go2arx5_vla.sla_adapter.provider_mfree_v1 import Provider
    from go2arx5_vla.sla_adapter.model_v1 import observation
    from go2arx5_vla.sla_adapter.codec_v1 import pad32,roundtrip_error
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from go2arx5_vla.world_model import fusion_trajectory_adapter as limits
    from go2arx5_vla.world_model.task_learning_executability_v3 import angular_velocity
    from scripts.pi0292.eval_contract import matrix10
    root=Path(s['registered_root']);C20=Path(s['C20_root']);pi0path=Path(inspect.getfile(pi0));assert sha(pi0path)==PI0_SHA
    fields,source=extracted_fields(batch_v2,pi0);(root/'EXTRACTED_SOURCE.private.py').write_text(source)
    actual_precision=jax.config.jax_default_matmul_precision;result=read(C20/'candidate1000/RESULT.private.json')
    dataset=MobileDataset(C20/'dataset',sha(C20/'dataset/MANIFEST.private.json'),resize_with_pad=resize_with_pad,max_cache_bytes=512*1024**2)
    bank=read(Path(s['previous_diagnostic'])/'WINDOW48_PLAN.private.json')['windows'];chosen=[]
    for family in ('F1','F2'):
        for phase in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):chosen.append(next(r for r in bank if r['family']==family and r['phase']==phase))
        chosen.extend([r for r in bank if r['family']==family and r['phase']=='CLOSE'and r['index']not in {c['index']for c in chosen}][:3])
    assert len(chosen)==16 and len({w['index']for w in chosen})==16
    write(root/'FIELD_PLAN.private.json',dict(windows=chosen,taus=[.1,.5,.9],noise_seed_base=2026111600,max_field_evaluations=240,max_prefix_builds=32,complete_sample_actions_calls=0,new_optimizer_updates=0,official_pi0_sha256=PI0_SHA,batch_error32_source_sha256=sha(inspect.getfile(batch_v2)),extracted_source_sha256=sha(root/'EXTRACTED_SOURCE.private.py'),actual_default_matmul_precision=actual_precision,highest_matmul_precision='highest',physical_outputs_are_single_field_clean_estimates_not_policy_rollouts=True))
    jax.config.update('jax_compilation_cache_dir',str(root/'cache'))
    write(root/'PROGRESS.private.json',dict(phase='LOAD_SAME_C20_1000_PROVIDER',epoch=time.time(),field_evaluations=0,complete_generations=0,new_updates=0))
    provider=Provider(result['package']['path'],manifest_sha256=result['package']['sha256'],parent_params=s['raw_parent'],parent_manifest_path=s['parent_manifest']['path']);graph=provider.graph
    def build(name,precision):
        def call(active,frozen,*args):
            with jax.default_matmul_precision(precision):return fields[name](nnx.merge(graph,active,frozen),*args)
        return jax.jit(call)
    train_high=build('train_field','highest');full_high=build('official_field','highest');prefix_high=build('prefix_field','highest');prefix_actual=build('prefix_field',actual_precision);cache_high=build('cache_field','highest');cache_actual=build('cache_field',actual_precision)
    def original_error(active,frozen,obs,actions,noise,time):
        with jax.default_matmul_precision('highest'):return batch_v2.error32(nnx.merge(graph,active,frozen),obs,actions,noise,time)
    error=jax.jit(original_error);rows=[];field_count=0;prefix_count=0
    for j,w in enumerate(chosen):
        index=w['index'];state,_,_,_=dataset.raw(index);ep,_=dataset.rows[index];frame=dataset.frames[ep['id']]['T_policy_from_mount'];codec=GripLogitCodec(frame);obs,actions=observation(dataset[index],provider.tokenizer)
        hi=prefix_high(provider.active,provider.frozen,obs);actual=prefix_actual(provider.active,provider.frozen,obs);jax.block_until_ready((hi,actual));prefix_count+=2
        for ti,tau in enumerate((.1,.5,.9)):
            noise=jax.random.normal(jax.random.PRNGKey(2026111600+j*3+ti),actions.shape);time_array=jnp.asarray([tau],jnp.float32);xt=time_array[:,None,None]*noise+(1-time_array[:,None,None])*actions
            values={}
            values['train_highest']=np.asarray(train_high(provider.active,provider.frozen,obs,xt,time_array));values['official_full_highest']=np.asarray(full_high(provider.active,provider.frozen,obs,xt,time_array));values['cached_highest']=np.asarray(cache_high(provider.active,provider.frozen,*hi,xt,jnp.asarray(tau,jnp.float32)));values['cached_actual']=np.asarray(cache_actual(provider.active,provider.frozen,*actual,xt,jnp.asarray(tau,jnp.float32)));field_count+=4
            e=np.asarray(error(provider.active,provider.frozen,obs,actions,noise,time_array));field_count+=1;expected=(values['train_highest']-(np.asarray(noise)-np.asarray(actions)))**2;error_identity=difference(e,expected)
            decoded={};kinematics={}
            for name,velocity in values.items():
                clean=np.asarray(xt)-np.float32(tau)*velocity;physical=codec.decode(state,pad32(provider.statistics['actions'].unnormalize(clean[0,:,:7])));decoded[name]=physical
                width=limits.WORKING_CLOSED_M+(limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*physical[:,9];speed=np.diff(width)/.05;acc=np.diff(np.r_[0.,speed,0.])/.05;kinematics[name]=dict(width_m=width.tolist(),finger_speed_m_s=speed.tolist(),finger_acceleration_m_s2=acc.tolist(),peak_finger_speed=float(abs(speed).max()),single_field_estimate_only=True)
            comparisons={}
            for left,right in (('train_highest','official_full_highest'),('official_full_highest','cached_highest'),('cached_highest','cached_actual')):
                a,b=decoded[left],decoded[right];delta_xyz=a[:,:3]-b[:,:3];delta_xyz_velocity=np.diff(delta_xyz,axis=0)/.05;delta_xyz_acceleration=np.diff(np.vstack((np.zeros(3),delta_xyz_velocity,np.zeros(3))),axis=0)/.05;delta_angular_velocity=angular_velocity(matrix10(a)[:,:3,:3])-angular_velocity(matrix10(b)[:,:3,:3]);physical_error=roundtrip_error(a,b);v=np.asarray(kinematics[left]['finger_speed_m_s'])-np.asarray(kinematics[right]['finger_speed_m_s']);acc=np.asarray(kinematics[left]['finger_acceleration_m_s2'])-np.asarray(kinematics[right]['finger_acceleration_m_s2']);comparisons[left+'__'+right]=dict(vector_field=difference(values[left],values[right]),clean_estimate_physical=physical_error,XYZ_speed_difference_max_m_s=float(np.linalg.norm(delta_xyz_velocity,axis=1).max()),XYZ_acceleration_difference_max_m_s2=float(np.linalg.norm(delta_xyz_acceleration,axis=1).max()),SO3_angular_speed_difference_max_rad_s=float(np.linalg.norm(delta_angular_velocity,axis=1).max()),finger_speed_difference_max_m_s=float(abs(v).max()),finger_acceleration_difference_max_m_s2=float(abs(acc).max()),finger_speed_gate_changed=(kinematics[left]['peak_finger_speed']>.04*(1+1e-10))!=(kinematics[right]['peak_finger_speed']>.04*(1+1e-10)))
            file=root/'data'/f'{index:04d}-t{ti}.npz'
            with file.open('xb')as h:np.savez(h,x_tau=np.asarray(xt),tau=np.float32(tau),**values)
            row=dict(window=w,tau=tau,comparisons=comparisons,actual_error32_vs_extracted_field_squared=error_identity,array_file=dict(path=str(file),sha256=sha(file)),kinematics=kinematics);rows.append(row)
            write(root/'PROGRESS.private.json',dict(phase='FIXED_VECTOR_FIELD_COMPARISON',epoch=time.time(),cases=len(rows),field_evaluations=field_count,prefix_builds=prefix_count,complete_generations=0,new_updates=0,new_physical_requests=0))
    write(root/'ALL_FIELD_COMPARISONS.private.json',rows);summary={}
    for key in rows[0]['comparisons']:
        c=[r['comparisons'][key]for r in rows];summary[key]=dict(max_active7_field_difference=max(x['vector_field']['active7_max_abs']for x in c),max_padding25_field_difference=max(x['vector_field']['padding25_max_abs']for x in c),max_clean_XYZ_difference_m=max(x['clean_estimate_physical']['xyz_m']for x in c),max_clean_SO3_difference_rad=max(x['clean_estimate_physical']['so3_rad']for x in c),max_clean_gripper_difference=max(x['clean_estimate_physical']['gripper']for x in c),max_XYZ_speed_difference_m_s=max(x['XYZ_speed_difference_max_m_s']for x in c),max_XYZ_acceleration_difference_m_s2=max(x['XYZ_acceleration_difference_max_m_s2']for x in c),max_SO3_angular_speed_difference_rad_s=max(x['SO3_angular_speed_difference_max_rad_s']for x in c),max_finger_speed_difference_m_s=max(x['finger_speed_difference_max_m_s']for x in c),max_finger_acceleration_difference_m_s2=max(x['finger_acceleration_difference_max_m_s2']for x in c),single_field_speed_gate_changes=sum(x['finger_speed_gate_changed']for x in c),all_vector_fields_bit_exact=all(x['vector_field']['bit_exact']for x in c))
    out=dict(schema='C20_VECTOR_FIELD_NUMERICAL_AUDIT_V3',epoch=time.time(),observations=16,tau_cases=len(rows),vector_field_evaluations=field_count,prefix_cache_builds=prefix_count,complete_sample_actions_calls=0,new_optimizer_updates=0,new_training_visits=0,new_physical_requests=0,comparisons=summary,original_error32_squared_max_abs=max(r['actual_error32_vs_extracted_field_squared']['max_abs']for r in rows),actual_default_matmul_precision=actual_precision,original_pinned_pi0_sha256=PI0_SHA,clean_estimate_scope='SINGLE_IDENTICAL_X_TAU_AND_TAU_NOT_A_POLICY_GENERATION_OR_NATIVE_CERTIFICATE',all_data=dict(path=str(root/'ALL_FIELD_COMPARISONS.private.json'),sha256=sha(root/'ALL_FIELD_COMPARISONS.private.json')))
    write(root/'RESULT.private.json',out);return out

if __name__=='__main__':
 s=read(sys.argv[sys.argv.index('--vector')+1]);root=Path(s['registered_root']);lock=OwnerLock(s['neural_owner_lock'])
 try:run(s);code=0
 except BaseException:write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_optimizer_updates=0,new_physical_requests=0));code=1
 finally:lock.close()
 from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
 close(s,code);raise SystemExit(code)
