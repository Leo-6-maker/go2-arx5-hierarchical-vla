"""Real shared codec checks, then fixed family/stage admission; CPU only."""
import copy
import json
import os
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
DATA=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/S2')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
sys.path[:0]=[str(DATA/'qa'),str(DATA/'runtime'),str(DATA/'R3'),'/home/liuyu-nfs/Projects/isaac-gr00t',
              str(BASE/'PI0250/runtime/reference-317a/src'),str(BASE/'PI0266/code')]
import pi0260_r2_codec as codec
from pi0260_r3_inputs import build,batch,identity,inventory
from pi0255_motion3 import sha,write
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
from pi0265_metrics import so3


def read(p):return json.loads(Path(p).read_text())


def roundtrip(statistics,population):
    proc=codec.processor(statistics);records=[]
    for family,state,action,rows in population:
        result=[]
        for row in rows:
            if not row['precodec_valid']:
                result.append(dict(index=row['index'],pass_codec=False,reason='PRECODEC_REJECT'));continue
            i=row['index'];packed_state,packed,mask=codec.encode(proc,state[i],action[i]);decoded=codec.decode(proc,packed,state[i])
            assert packed.shape==(40,132) and packed_state.shape==(1,132) and np.all(mask[:,:10]==1) and np.all(mask[:,10:]==0)
            xyz=float(np.max(np.linalg.norm(decoded[:,:3]-action[i,:,:3],axis=-1)))
            angle=float(np.max(so3(rot6d_rows_to_matrix(decoded[:,3:9]),rot6d_rows_to_matrix(action[i,:,3:9]))))
            result.append(dict(index=i,pass_codec=xyz<=1e-5 and angle<=1e-4,XYZ_max_m=xyz,SO3_max_rad=angle,
                destructive_clipping=False))
        records.append((family,result))
    return records


def main():
    card=read(ROOT/'design/SCENE_CARD.json');families=[f for f in card['families'] if f['split']=='FEEDBACK_TRAIN']
    assert len(families)==24 and all((RAW/f['family_id']/'COMPLETE.json').exists() for f in families)
    assert not (R1/'design/DATA_ADMISSION.json').exists(),'sealed admission retained'
    population=[]
    for f in families:
        folder=RAW/f['family_id'];complete=read(folder/'COMPLETE.json')
        assert all(sha(folder/n)==h for n,h in complete['hashes'].items())
        z=np.load(folder/'proposals.npz');rows=read(folder/'rows.json')
        population.append((f['family_id'],z['state_v1'].copy(),z['raw_action'].copy(),rows))
    old=read(DATA/'dataset/statistics.json');assert sha(DATA/'dataset/statistics.json')=='4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73'
    checks=roundtrip(old,population);adapted=any(not q['pass_codec'] for fid,rs in checks for q in rs if q.get('reason')!='PRECODEC_REJECT')
    statistics=copy.deepcopy(old)
    if adapted:
        # The only permitted adaptation: one pooled TRAIN union, fixed XYZ padding.
        valid=[(s[r['index']],a[r['index']]) for f,s,a,rs in population for r in rs if r['precodec_valid']]
        states=np.stack([s for s,a in valid]);actions=np.stack([a for s,a in valid])
        relatives=np.stack([codec.relative(a,s) for s,a in valid])
        for section,values in [('state',states[:,:3]),('action',actions[:,:,:3].reshape(-1,3)),('relative_action',relatives[:,:,:3].reshape(-1,3))]:
            lo=values.min(0);hi=values.max(0);span=np.maximum(hi-lo,.001);mid=(lo+hi)/2;lo=mid-span*.55;hi=mid+span*.55
            params=statistics[codec.TAG][section]['eef_9d']
            for key,values in [('min',lo),('q01',lo),('max',hi),('q99',hi)]:params[key][:3]=values.tolist()
        checks=roundtrip(statistics,population)
    write(R1/'design/statistics.json',statistics)
    admitted=[];family_results=[]
    for (fid,states,actions,rows),(other,rs) in zip(population,checks):
        assert fid==other;valid={q['index'] for q in rs if q['pass_codec']};strata=[];selected=[]
        for task in ('M1','M3'):
            for command in (0,1):
                own=[r for r in rows if r['task']==task and r['command']==command]
                nominal=[r for r in own if r['nominal'] and r['index'] in valid]
                quarters=[[r['anchor_index'] for r in nominal if not r['terminal'] and r['quartile']==q] for q in range(4)]
                terminal=any(r['terminal'] for r in nominal)
                missing=[r['anchor_index'] for r in nominal if not r['terminal'] and not any(v['anchor_index']==r['anchor_index'] and not v['nominal'] and v['index'] in valid for v in own)]
                good=all(len(q)>=3 for q in quarters) and terminal and not missing
                strata.append(dict(task=task,command=command,nominal_quartiles=quarters,terminal=terminal,missing_recovery=missing,pass_admission=good))
                selected.extend(r['index'] for r in own if r['index'] in valid)
        good=all(x['pass_admission'] for x in strata)
        if good:admitted.append(fid)
        item=dict(family=fid,pass_admission=good,strata=strata,eligible_indices=selected,codec=rs)
        write(RAW/fid/'codec_admission.json',item);family_results.append(item)
    receipt=dict(schema='PI0267_VIRTUAL_DATA_ADMISSION_V1',status='PASS_CONTINUE_ACTUAL_INPUT_AND_TRAINING' if len(admitted)>=16 else 'PRIMARY_PARTIAL_DATA_INTEGRITY_FAIL',
        families_admitted=admitted,families_total=24,proposals_total=10752,unique_policy_frame_pairs=sum(read(RAW/f['family_id']/'COMPLETE.json')['unique_frame_pairs'] for f in families),
        statistics_adapted_once=adapted,old_statistics_sha256=sha(DATA/'dataset/statistics.json'),statistics_sha256=sha(R1/'design/statistics.json'),
        teacher_sha256=sha(ROOT/'code/pi0267_expert.py'),tool_card_sha256=sha(R1/'design/VIRTUAL_TOOL_CARD.json'),scene_card_sha256=sha(ROOT/'design/SCENE_CARD.json'),
        model_updates=0,model_queries=0,EVAL_opened=False,physics_steps=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
        max_roundtrip_XYZ_m=max(q.get('XYZ_max_m',0) for fid,rs in checks for q in rs),max_roundtrip_SO3_rad=max(q.get('SO3_max_rad',0) for fid,rs in checks for q in rs),
        summaries=[{k:v for k,v in f.items() if k not in ('codec','eligible_indices')} for f in family_results])
    write(R1/'design/DATA_ADMISSION.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
