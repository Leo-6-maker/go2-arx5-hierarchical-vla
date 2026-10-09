"""Bounded explanation of the 15 sealed failed H40 windows; no admission change."""
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '4'
ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
sys.path[:0] = [str(ROOT/'code'), str(ROOT.parent/'PI0283/code'),
               os.environ.get('PI0282_OFFICIAL_DIR',str(ROOT.parent/'PI0280/official'))]
import numpy as np
import torch
torch.set_num_threads(4)
from data import codec
from recovery import FROZEN
from gr00t.data.types import ActionType, ActionFormat
from gr00t.data.utils import normalize_values_minmax

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

started = time.time()
data = ROOT/'data'
bank_path = ROOT/'correction/CORRECTION_BANK.json'
bank = json.loads(bank_path.read_text())
assert bank['status'] == 'CORRECTION_NOT_ADMITTED' and bank['valid_branches'] == 15
reports = [r for r in bank['all_requested_slot_reports'] if r['status']=='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH']
assert len(reports)==13 and sum(len(r['failures']) for r in reports)==15
paths = {bank_path, ROOT/'design/CODE_SEAL.json'} | {data/k for k in FROZEN}
for r in reports:
    slot=r['slot']
    paths |= {data/'action_target'/slot/'action.npy',data/'policy_observation'/slot/'state_v1.npy'}
before={str(p):sha(p) for p in paths}
assert {k:before[str(data/k)] for k in FROZEN}==FROZEN==bank['processor']
encoder=codec(str(data))
assert encoder.identity==FROZEN
p=encoder.proc.state_action_processor
tag='new_embodiment'
param=p.norm_params[tag]['action']
names=['relative_x','relative_y','relative_z']+[f'relative_rot6d_{i}' for i in range(6)]+['absolute_gripper']
windows=[]
for r in reports:
    slot=r['slot']
    action_path=data/'action_target'/slot/'action.npy'
    state_path=data/'policy_observation'/slot/'state_v1.npy'
    for path in (action_path,state_path):
        key=str(path.relative_to(data))
        assert before[str(path)]==r['rows'][0]['source_identity'][key]
    actions=np.load(action_path,mmap_mode='r',allow_pickle=False)
    states=np.load(state_path,mmap_mode='r',allow_pickle=False)
    for failure in r['failures']:
        a=failure['anchor']; target=actions[a:a+40].copy();state=states[a].copy()
        assert target.shape==(40,10) and state.shape==(10,)
        relative=p._convert_to_relative_action(target[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
        native=np.concatenate([relative,target[:,9:10]],axis=-1)
        raw=np.concatenate([normalize_values_minmax(relative,param['eef_9d']),
                            normalize_values_minmax(target[:,9:10],param['gripper_position'])],axis=-1)
        audit=encoder.action_audit(target,state)
        count=int((np.abs(raw)>1).sum())
        assert count==audit['native_preclip_out_of_range_count']==failure['errors']['native_preclip_out_of_range_count']
        channels=[]
        for i,name in enumerate(names):
            modality='eef_9d' if i<9 else 'gripper_position';j=i if i<9 else 0
            bounds=param[modality];low=float(bounds['min'][j]);high=float(bounds['max'][j])
            x=raw[:,i];v=native[:,i];outside=np.abs(x)>1
            channels.append(dict(channel=i,name=name,source_modality=modality,out_of_range_count=int(outside.sum()),
                below_minus_one_count=int((x < -1).sum()),above_plus_one_count=int((x > 1).sum()),
                normalized_min=float(x.min()),normalized_max=float(x.max()),max_normalized_excess=float(np.maximum(np.abs(x)-1,0).max()),
                native_min=float(v.min()),native_max=float(v.max()),frozen_TRAIN_min=low,frozen_TRAIN_max=high,
                frozen_TRAIN_width=high-low,max_native_below_TRAIN_min=float(max(0,low-float(v.min()))),
                max_native_above_TRAIN_max=float(max(0,float(v.max())-high))))
        windows.append(dict(slot=slot,anchor=a,original_control_tick=r['takeover_tick']+a,
            source_sha256={str(q.relative_to(data)):before[str(q)] for q in (action_path,state_path)},
            target_dtype=str(target.dtype),state_dtype=str(state.dtype),relative_dtype=str(relative.dtype),
            preclip_dtype=str(raw.dtype),native_dtype=str(native.dtype),sealed_count=count,
            actual_action_audit=audit,channels=channels,
            sealed_roundtrip_errors=failure['errors']))
    del actions,states
after={str(q):sha(q) for q in paths}
assert before==after
functions={'normalize_values_minmax':normalize_values_minmax,'_convert_to_relative_action':p._convert_to_relative_action,'action_audit':encoder.action_audit}
sources={n:dict(file=inspect.getsourcefile(f),sha256=sha(inspect.getsourcefile(f)),
               function_source_sha256=hashlib.sha256(inspect.getsource(f).encode()).hexdigest()) for n,f in functions.items()}
report=dict(status='PASS_BOUNDED_PRECLIP_EXPLANATION_COUNTS_AND_HASHES_NOT_ADMISSION',
    windows=len(windows),failed_branches=len(reports),frozen_identity=FROZEN,
    input_and_statistics_hashes_before=before,input_and_statistics_hashes_after=after,
    code_functions=sources,normalization_source=inspect.getsource(normalize_values_minmax),
    window_details=windows,admission_status_unchanged=bank['status'],valid_branches_unchanged=15,
    model_calls=0,teacher_calls=0,physics_calls=0,full_codec_calls=0,
    examined_saved_H40_windows=15,raw_slot_files_opened=0,elapsed_seconds=time.time()-started)
out=ROOT/'correction/PRECLIP_FAILED15_DETAILS.json'
assert not out.exists(),'Do not overwrite a closed explanation'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],windows=15,output=str(out),sha256=sha(out),elapsed_seconds=report['elapsed_seconds'])))
