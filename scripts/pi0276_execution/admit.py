"""New TRAIN-only task affine bounds on the existing relative-SE3 codec."""
import os,sys,json,ast
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
    OPENBLAS_CORETYPE='Prescott',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0276';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')
OLD=RAW.parent/'PI0260/R2'
sys.path[:0]=[str(OLD/'R3'),str(OLD/'runtime'),str(OLD/'qa'),str(BASE/'PI0250/runtime/reference-317a/src'),
    str(BASE/'PI0250/runtime/reference-317a/scripts'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(ROOT/'project-src')]
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pi0260_r2_codec as codec
from pi0260_r3_inputs import build,batch,identity
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
from run import sha,write
from dataset import freeze

def calibrated(values,axis,minimum=None,rotation=False,command=False):
    actual=codec.stats(values,axis);lo=np.asarray(actual['min']);hi=np.asarray(actual['max'])
    if rotation:
        assert np.max(np.abs(values[...,3:9]))<=1.00001
        lo[...,3:9]=-1;hi[...,3:9]=1
        width=np.maximum(hi[...,:3]-lo[...,:3],.10);mid=(hi[...,:3]+lo[...,:3])/2
        lo[...,:3]=mid-width/2;hi[...,:3]=mid+width/2
    elif command:lo[:]=0;hi[:]=1
    else:
        width=np.maximum(hi-lo,minimum);mid=(hi+lo)/2;lo=mid-width/2;hi=mid+width/2
    result=dict(actual,q01=lo.tolist(),q99=hi.tolist())
    return result,actual

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    original=torch.nn.Module.__call__
    def no_model(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    torch.nn.Module.__call__=no_model
    card=json.loads((ROOT/'design/DATASET_CARD.json').read_text());allowed={(e['split'],e['seed']) for e in card['episodes'] if e['admitted']}
    bank=[r for r in card['rows'] if r['eligible'] and (r['split'],r['seed']) in allowed]
    states=[];actions=[];relatives=[]
    for e in card['episodes']:
        if not e['admitted'] or e['split']!='TRAIN':continue
        folder=Path(e['path'])
        with np.load(folder/'observations.npz') as z:s=z['state_v1'].copy()
        with np.load(folder/'commands.npz') as z:a=z['action'].copy()
        for r in bank:
            if r['split']!='TRAIN' or r['seed']!=e['seed']:continue
            j=r['anchor'];state=s[j];action=a[j:j+40].astype(np.float32)
            relative=codec.relative(action,state)
            rs=rot6d_rows_to_matrix(relative[:,3:9])
            assert np.max(np.abs(relative[:,3:9]-rs[:,:2,:].reshape(40,6)))<1e-5, 'Relative boundary is not physical row6d'
            states.append(state);actions.append(action);relatives.append(relative)
    states=np.asarray(states);actions=np.asarray(actions);relatives=np.asarray(relatives)
    assert states[:,9].ptp()>1e-4 and np.min(actions[:,:,9])<=1e-6 and np.max(actions[:,:,9])>=1-1e-6
    stats={'new_embodiment':dict(state={},action={},relative_action={})};extrema={}
    for category,key,values,axis,kw in [
        ('state','eef_9d',states[:,:9],0,dict(rotation=True)),
        ('state','gripper_position',states[:,9:],0,dict(minimum=.10)),
        ('action','eef_9d',actions[:,:,:9],(0,1),dict(rotation=True)),
        ('action','gripper_position',actions[:,:,9:],(0,1),dict(command=True)),
        ('relative_action','eef_9d',relatives,0,dict(rotation=True))]:
        params,actual=calibrated(values,axis,**kw);stats['new_embodiment'][category][key]=params;extrema[category+'/'+key]=actual
    statfile=ROOT/'contracts/TASK/source/dataset/statistics.json'
    freeze(statfile,stats)
    freeze(ROOT/'design/PROCESSOR_CONTRACT.json',dict(schema='PI0276_TASK_SE3_NORMALIZATION_V1',TRAIN_only=True,
        affine_calibration_fields=['q01','q99'],fields_are_not_empirical_quantiles=True,
        physical_row6d_boundary_proved=True,actual_extrema_true_quantiles=extrema,statistics_sha256=sha(statfile)))
    proc=build(stats,BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot');rawproc=codec.processor(stats);proof=[]
    # Reuse the executed gate adapter without importing or starting its physics runtime.
    gate_source=ROOT/'code/source_worker.py'
    node=next(n for n in ast.parse(gate_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pose_gate')
    scope=dict(ROOT=ROOT,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(gate_source),'exec'),scope)
    pose_gate=scope['pose_gate']
    for e in card['episodes']:
        if not e['admitted']:continue
        folder=Path(e['path'])
        with np.load(folder/'observations.npz') as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        with np.load(folder/'commands.npz') as z:commands=z['action'].copy().astype(np.float32)
        for r in [r for r in bank if r['split']==e['split'] and r['seed']==e['seed']]:
            j=r['anchor'];state=obs['state_v1'][j];action=commands[j:j+40]
            s,a,mask=codec.encode(rawproc,state,action);decoded=codec.decode(rawproc,a,state)
            xyz=float(np.linalg.norm(decoded[:,:3]-action[:,:3],axis=-1).max());grip=float(np.max(np.abs(decoded[:,9]-action[:,9])))
            expected=rot6d_rows_to_matrix(action[:,3:9]);actual=rot6d_rows_to_matrix(decoded[:,3:9])
            angle=float(Rotation.from_matrix(actual@expected.transpose(0,2,1)).magnitude().max())
            assert xyz<=1e-5 and angle<=1e-4 and grip<=1e-6,(r['uid'],xyz,angle,grip)
            assert s.shape==(1,132) and a.shape==mask.shape==(40,132) and not a[:,10:].any() and not s[:,10:].any()
            assert np.all(mask[:,:10]==1) and not mask[:,10:].any()
            packet=dict(body_rgb=obs['base_camera_96'][j],wrist_rgb=obs['hand_camera_96'][j],state_v1=state,
                language='Stack the red cube on the green cube and release it.')
            value=batch(proc,packet,action)['inputs'];condition={k:v for k,v in value.items() if k not in ('action','action_mask')}
            assert identity(condition)==identity(batch(proc,packet)['inputs'])
            assert value['state'].dtype==value['action'].dtype==torch.float32
            before=pose_gate(action,state,j*.05,r['uid']);after=pose_gate(decoded,state,j*.05,r['uid'])
            assert before['decision']==after['decision'], ('CODEC_GATE_DISCREPANCY',r['uid'])
            proof.append(dict(uid=r['uid'],condition_sha256=identity(condition),labels_masks_sha256=identity({k:value[k] for k in ('action','action_mask')}),
                XYZ_m=xyz,SO3_rad=angle,gripper=grip,normalized_state_abs_max=float(np.abs(s).max()),normalized_action_abs_max=float(np.abs(a).max()),gate=before['decision']))
        print('ACTUAL_TASK_CODEC',e['split'],e['seed'],len(proof),flush=True)
    freeze(ROOT/'design/PROCESSED_INPUTS.json',dict(rows=proof,statistics_sha256=sha(statfile)))
    write(ROOT/'evidence/DATA_ADMISSION.json',dict(status='PASS_PI0276_ACTUAL_REPLAY_INPUTS_CODEC_AND_TASK_STATISTICS',launch_allowed=True,
        conditions=len(proof),statistics_sha256=sha(statfile),dataset_card_sha256=sha(ROOT/'design/DATASET_CARD.json'),train_plan_sha256=sha(ROOT/'design/TRAIN_PLAN.json'),
        modules=[dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__)) for m in (codec,sys.modules['pi0260_r3_inputs'],sys.modules['pi0260_r2_inputs'])],
        model_queries=0,optimizer_updates=0,new_physics_steps=0))

if __name__=='__main__':main()
