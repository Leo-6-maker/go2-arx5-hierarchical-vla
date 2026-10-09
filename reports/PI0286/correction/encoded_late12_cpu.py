"""Inspect actual native encode/decode dtypes and clipping for 12 sealed late windows."""
import hashlib,inspect,json,os,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
sys.path[:0]=[str(ROOT/'code'),str(ROOT.parent/'PI0283/code'),str(ROOT.parent/'PI0280/official')]
import numpy as np,torch
torch.set_num_threads(4)
from data import codec
from recovery import FROZEN
from cpu_inputs import physical_errors
from gr00t.data.types import ActionType,ActionFormat
from gr00t.data.utils import normalize_values_minmax,unnormalize_values_minmax
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ranges(x):
    x=np.asarray(x)
    return dict(dtype=str(x.dtype),shape=list(x.shape),min=float(x.min()),max=float(x.max()),outside_count=int((np.abs(x)>1).sum()))
started=time.time();data=ROOT/'data';bp=ROOT/'correction/CORRECTION_BANK.json'
bank=json.loads(bp.read_text());reports=[r for r in bank['all_requested_slot_reports'] if r['status']=='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH' and r['takeover_tick']==260]
assert len(reports)==12 and all(len(r['failures'])==1 and r['failures'][0]['anchor']==0 for r in reports)
paths={bp,ROOT/'design/CODE_SEAL.json'}|{data/k for k in FROZEN}
for r in reports:paths|={data/'action_target'/r['slot']/'action.npy',data/'policy_observation'/r['slot']/'state_v1.npy'}
before={str(p):sha(p) for p in paths};assert {k:before[str(data/k)] for k in FROZEN}==FROZEN
encoder=codec(str(data));assert encoder.identity==FROZEN
p=encoder.proc.state_action_processor;params=p.norm_params['new_embodiment']['action']
functions={'apply_action':p.apply_action,'unapply_action':p.unapply_action,'encode_actions':encoder.encode_actions,'decode_actions':encoder.decode_actions,'action_audit':encoder.action_audit,'normalize_values_minmax':normalize_values_minmax,'unnormalize_values_minmax':unnormalize_values_minmax,'_convert_to_relative_action':p._convert_to_relative_action}
sources={n:dict(file=inspect.getsourcefile(f),file_sha256=sha(inspect.getsourcefile(f)),line=inspect.getsourcelines(f)[1],source=inspect.getsource(f)) for n,f in functions.items()}
orig_clip=np.clip;orig_apply=p.apply_action;orig_unapply=p.unapply_action
events=[];stage='';outputs=[]
def observed_clip(a,*args,**kwargs):
    x=np.asarray(a).copy();result=orig_clip(a,*args,**kwargs);y=np.asarray(result)
    caller=inspect.currentframe().f_back
    events.append(dict(stage=stage,function=caller.f_code.co_name,file=caller.f_code.co_filename,line=caller.f_lineno,
        input=ranges(x),output=ranges(y),changed_value_count=int((x!=y).sum()),max_abs_change=float(np.max(np.abs(x-y),initial=0))))
    return result
def observed_apply(*args,**kwargs):
    result=orig_apply(*args,**kwargs);outputs.append(dict(stage=stage,method='apply_action',outputs={k:ranges(v) for k,v in result.items()}));return result
def observed_unapply(*args,**kwargs):
    result=orig_unapply(*args,**kwargs);outputs.append(dict(stage=stage,method='unapply_action',outputs={k:ranges(v) for k,v in result.items()}));return result
windows=[]
try:
    np.clip=observed_clip;p.apply_action=observed_apply;p.unapply_action=observed_unapply
    for r in reports:
        slot=r['slot'];ap=data/'action_target'/slot/'action.npy';sp=data/'policy_observation'/slot/'state_v1.npy'
        for path in (ap,sp):assert before[str(path)]==r['rows'][0]['source_identity'][str(path.relative_to(data))]
        target=np.load(ap,mmap_mode='r',allow_pickle=False)[:40].copy();state=np.load(sp,mmap_mode='r',allow_pickle=False)[0].copy()
        relative=p._convert_to_relative_action(target[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
        raw=np.concatenate([normalize_values_minmax(relative,params['eef_9d']),normalize_values_minmax(target[:,9:10],params['gripper_position'])],axis=-1)
        audit=encoder.action_audit(target,state);assert audit['native_preclip_out_of_range_count']==80==r['failures'][0]['errors']['native_preclip_out_of_range_count']
        events=[];outputs=[];stage='ACTUAL_ENCODE';encoded=encoder.encode_actions(target,state)
        stage='ACTUAL_DECODE';decoded=encoder.decode_actions(encoded,state)
        stage='STORED_THRESHOLD_DIAGNOSTIC';error=physical_errors(target,decoded)
        channels=[]
        for i in range(10):
            channels.append(dict(channel=i,preclip64_min=float(raw[:,i].min()),preclip64_max=float(raw[:,i].max()),
                preclip64_outside_count=int((np.abs(raw[:,i])>1).sum()),encoded32_min=float(encoded[:,i].min()),encoded32_max=float(encoded[:,i].max()),
                encoded32_outside_count=int((np.abs(encoded[:,i])>1).sum()),
                actual_encoded_vs_direct_preclip_FP32_cast_max_abs_diff=float(np.max(np.abs(encoded[:,i]-raw[:,i].astype(np.float32)),initial=0))))
        windows.append(dict(slot=slot,anchor=0,original_tick=260,target=ranges(target),state=ranges(state),preclip=ranges(raw),
            encoded=ranges(encoded),encoded_active10=ranges(encoded[:,:10]),decoded=ranges(decoded),
            sealed_preclip_count=80,actual_native_action_audit=audit,clip_outliers=p.clip_outliers,
            actual_clip_calls=events,actual_processor_outputs=outputs,channels=channels,
            roundtrip_errors=error,
            physical_thresholds_satisfied=all(error[k]<=t for k,t in [('XYZ_max_m',1e-5),('SO3_max_rad',1e-4),('gripper_max_fraction',1e-5)]),
            direct_preclip_FP32_cast_equals_actual_encoded_active10=bool(np.array_equal(raw.astype(np.float32),encoded[:,:10]))))
finally:
    np.clip=orig_clip;p.apply_action=orig_apply;p.unapply_action=orig_unapply
after={str(p):sha(p) for p in paths};assert before==after
assert all(sha(v['file'])==v['file_sha256'] for v in sources.values())
result=dict(status='ACTUAL_NATIVE_ENCODE_CLIP_DTYPE_EVIDENCE_OWNER_CONTRACT_DECISION_PENDING',windows=12,
    bounded_anchors='ONLY_12_PREVIOUSLY_FAILED_LATE_ANCHOR0',frozen_identity=FROZEN,
    hashes_before=before,hashes_after=after,native_functions=sources,
    statistics_parameter_dtypes={k:{n:str(np.asarray(v).dtype) for n,v in bounds.items()} for k,bounds in params.items()},
    window_details=windows,admission_state_written=False,labels_written=False,statistics_changed=False,
    model_calls=0,teacher_calls=0,physics_calls=0,actual_encode_calls=12,actual_decode_calls=12,
    raw_slot_files_opened=0,elapsed_seconds=time.time()-started)
out=ROOT/'correction/ENCODED_LATE12_NATIVE_CLIP_EVIDENCE.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],windows=12,sha256=sha(out),elapsed_seconds=result['elapsed_seconds'],
    all_encoded_active_outside_counts=[w['encoded_active10']['outside_count'] for w in windows],
    all_direct_FP32_cast_equal=all(w['direct_preclip_FP32_cast_equals_actual_encoded_active10'] for w in windows),
    all_physical_thresholds_satisfied=all(w['physical_thresholds_satisfied'] for w in windows))))
