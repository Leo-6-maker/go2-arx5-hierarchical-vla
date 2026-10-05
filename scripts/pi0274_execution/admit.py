"""Unchanged actual processor/codec at all registered poses and supplemental languages."""
import os,sys,json,ctypes,resource
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0274';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274')
DATA=RAW.parent/'PI0260/R2'
sys.path[:0]=[str(DATA/'R3'),str(DATA/'runtime'),str(DATA/'qa'),str(BASE/'PI0250/runtime/reference-317a/src'),str(BASE/'PI0250/runtime/reference-317a/scripts'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(BASE/'PI0266/code'),str(BASE/'PI0273/code')]
import numpy as np
import torch
import pi0260_r2_codec as codec
from pi0260_r3_inputs import build,batch,identity
from quality_gate import read,sha
from pi0255_motion3 import write
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
from pi0265_metrics import so3

def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original=torch.nn.Module.__call__
    def no_model(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original(self,*a,**kw)
        raise RuntimeError('PI0274_CPU_NO_LEARNED_CALL')
    torch.nn.Module.__call__=no_model
    def no_optimizer(*a,**kw):raise RuntimeError('PI0274_CPU_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=no_optimizer;assert not torch.cuda.is_available()
    pre=read(ROOT/'evidence/DATA_PRECODEC.json');assert pre['total_proposals']==1248
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    core=ctypes.CDLL(str(library)).openblas_get_corename64_;core.restype=ctypes.c_char_p;assert core().decode()=='Prescott'
    stats=BASE/'PI0273/contracts/ROT/source/dataset/statistics.json';assert sha(stats)=='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    proc=build(read(stats),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot');rawproc=codec.processor(read(stats))
    card=read(ROOT/'design/POSE_CARD.json');plan=read(ROOT/'design/TRAIN_PLAN.json');checks=[];inputs={};failures=[];normalized={}
    by={r['id']:r for r in card['rows']};needed={r['id']:{r['field_language']} for r in card['rows']}
    for visit in plan['stream']:
        if visit['stream']=='SUPPLEMENT':needed[visit['pose_id']].add(visit['language'])
    for i,row in enumerate(card['rows']):
        out=RAW/'data'/row['id'];receipt=read(out/'receipt.json');assert receipt['proposal']==row and receipt['sample_sha256']==sha(out/'sample.npz')
        with np.load(out/'sample.npz') as z:sample={k:z[k].copy() for k in z.files}
        state,action=sample['state_v1'],sample['raw_action'];encoded,packed,mask=codec.encode(rawproc,state,action);decoded=codec.decode(rawproc,packed,state)
        xyz=float(np.max(np.linalg.norm(decoded[:,:3]-action[:,:3],axis=-1)));angle=float(np.max(so3(rot6d_rows_to_matrix(decoded[:,3:9]),rot6d_rows_to_matrix(action[:,3:9]))))
        valid=receipt['valid'] and xyz<=1e-5 and angle<=1e-4
        checks.append(dict(id=row['id'],training_permitted=row['training_permitted'],precodec_valid=receipt['valid'],XYZ_max_m=xyz,SO3_max_rad=angle,valid=valid,reasons=receipt['reasons']+([] if xyz<=1e-5 and angle<=1e-4 else ['CODEC_ROUNDTRIP'])))
        if row['training_permitted'] and not valid:failures.append(checks[-1])
        assert encoded.shape==(1,132) and packed.shape==(40,132) and np.isfinite(packed).all() and np.isfinite(encoded).all()
        assert not packed[:,10:].any() and np.all(mask[:,:10]==1) and not mask[:,10:].any()
        own=[]
        for language in sorted(needed[row['id']]):
            packet={k:sample[k] for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=language
            value=batch(proc,packet,action)['inputs'];condition={k:v for k,v in value.items() if k not in ('action','action_mask')};labels={k:value[k] for k in ('action','action_mask')}
            assert value['state'].shape==(1,1,132) and value['action'].shape==(1,40,132)
            assert value['state'].dtype==value['action'].dtype==torch.float32 and torch.isfinite(value['state']).all() and torch.isfinite(value['action']).all()
            assert not value['state'][...,10:].any() and not value['action'][...,10:].any()
            for name in ('state','action'):normalized[name]=max(normalized.get(name,0.),float(value[name].abs().max()))
            item=dict(pose_id=row['id'],language=language,condition_sha256=identity(condition),labels_masks_sha256=identity(labels),sample_sha256=receipt['sample_sha256'],path=str(out/'sample.npz'),valid=valid)
            inputs[row['id'],language]=item;own.append(item)
        write(out/'processed_inputs.json',own)
        if (i+1)%104==0:print('ACTUAL_CODEC_AND_INPUTS',i+1,len(inputs),len(failures),flush=True)
    stream=[]
    for visit in plan['stream']:
        v=dict(visit)
        if v['stream']=='SUPPLEMENT':v.update(inputs[v['pose_id'],v['language']])
        stream.append(v)
    assert len(stream)==10240 and len({v['pose_id'] for v in stream if v['stream']=='SUPPLEMENT'})==672
    write(ROOT/'design/FITTING_CARD.json',dict(schema='PI0274_ACTUAL_GOAL_LOCAL_SUPPLEMENT_V1',base=plan['base'],streams=dict(EQUIL=stream),base_source_card_sha256=plan['source_card_sha256'],new_inputs=list(inputs.values()),
        field_inputs=[inputs[r['id'],r['field_language']] for r in card['rows']],source_pose_card_sha256=sha(ROOT/'design/POSE_CARD.json'),statistics_sha256=sha(stats),updates_per_arm=5120,training_seed=27303))
    write(ROOT/'evidence/DATA_ADMISSION.json',dict(status='DATA_INCOMPLETE' if failures else 'PASS_ALL672_REAL_POSES_CODEC_INPUTS_AND_FIXED_SCHEDULE',
        proposals=1248,training_required=672,training_failures=failures,checks=checks,actual_processed_conditions=len(inputs),base_encoding_proof_reused=str(BASE/'PI0273/evidence/training-proofs/evidence/BASE_ENCODING8192.json'),
        base_encoding_proof_sha256=sha(BASE/'PI0273/evidence/training-proofs/evidence/BASE_ENCODING8192.json'),normalized_abs_max=normalized,statistics_sha256=sha(stats),
        max_XYZ_roundtrip_m=max(r['XYZ_max_m'] for r in checks),max_SO3_roundtrip_rad=max(r['SO3_max_rad'] for r in checks),kernel='Prescott',library_sha256=sha(library),
        launch_allowed=not failures,model_queries=0,optimizer_updates=0,teacher_calls=0,renders=0,statistics_refit=False))
    print('ACTUAL_DATA_ADMISSION',len(failures),'training failures',flush=True)

if __name__=='__main__':main()
