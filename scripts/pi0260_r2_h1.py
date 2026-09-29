"""Seal actual collection and qualify its one-fit real-codec dataset on CPU."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pi0260_r2 import ROOT,read,write,sha,admission,tier,post_codec
import pi0260_r2_codec as codec
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.safety import TrajectorySafetyGateV1
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

def seal(commit):
    admission()
    assert read(ROOT/'state/collection_owner.json')['status']=='COLLECTION_COMPLETE_REPLAY_CODEC_PENDING'
    files=[]
    for d in ('design','proposals'):
        files += [p for p in (ROOT/d).rglob('*') if p.is_file()]
    files += [ROOT/'state'/n for n in ('design_freeze.json','selection_TRAIN.json','selection_KIN-VAL.json','launch.json','collection_owner.json')]
    files += list((ROOT/'runtime').glob('*.py'))+list((ROOT/'qa').rglob('*.py'))+list((ROOT/'evidence').glob('*.json'))
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    manifest=ROOT/'handoff/H1_geometry/manifest.json'
    write(manifest,dict(schema='PI0260_R2_COLLECTION_V1',code_commit=commit,entries=entries,authority=read(ROOT/'design/overlay.json')['authority'],
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(manifest=str(manifest),sha256=sha(manifest),files=len(entries))),flush=True)

def windows(reg,seq):
    folder=ROOT/'proposals'/seq['family_id'];base=folder/seq['sequence_id']
    with np.load(base/'geometry.npz') as z:a={k:z[k].copy() for k in z.files}
    raw={r['index']:r for r in read(base/'raw_gate.json')['rows']}
    obs=read(base/'observations.json')['rows'];anchor=np.asarray(reg['initial_TCP_world'])[:3,3]
    for o in obs:
        i=o['index'];ii=np.minimum(i+np.arange(40),len(a['joints'])-1)
        frame=folder/'frames'/(o['frame']+'.npz');assert sha(frame)==o['array_sha256']
        with np.load(frame) as z:state=z['state_v1'].astype(np.float32)
        label=np.c_[a['desired_position'][ii]-anchor,a['desired_rotation'][ii,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
        yield dict(index=i,time=o['time'],active=o['active'],frame=str(frame.relative_to(ROOT)),frame_sha256=o['array_sha256'],
            raw_good=raw[i]['verdict']['decision']!='REJECT',visible=o['check']['pass_necessary']),state,label

def build():
    reg=admission();qa=read(ROOT/'review_geometry/receipt.json')
    assert qa['status']=='PASS_R2_COLLECTION_READONLY_REPLAY'
    assert qa['producer_manifest_sha256']==sha(ROOT/'handoff/H1_geometry/manifest.json')
    selected=qa['selected_by_split'];pop=[s for s in reg['sequences'] if s['family_id'] in selected['TRAIN']]
    statpath=ROOT/'dataset/statistics.json';pop_path=ROOT/'dataset/statistics_population.json'
    if not statpath.exists():
        assert not pop_path.exists(),'incomplete one-fit statistics needs source reconciliation'
        states=[];actions=[];relative=[];sources=[]
        for seq in pop:
            for row,state,label in windows(reg,seq):
                if not(row['active'] and row['raw_good']):continue
                states.append(state);actions.append(label);relative.append(codec.relative(label,state))
                sources.append(dict(sequence=seq['sequence_id'],index=row['index']))
        statistics=codec.fit(states,actions,relative)
        write(statpath,statistics)
        write(pop_path,dict(id=codec.IDENTITY,statistics_sha256=sha(statpath),fit_count=1,selected_pre_codec_TRAIN=selected['TRAIN'],
            source_windows=sources,source_rule='all raw-gate-qualified active windows of fixed selected TRAIN; before codec, no VAL/DEV/TEST',
            relative_shape=[40,9],q_method='numpy linear',range_floor=1e-6,clamp=False))
    stats_pop=read(pop_path);assert stats_pop['fit_count']==1 and stats_pop['statistics_sha256']==sha(statpath)
    assert stats_pop['selected_pre_codec_TRAIN']==selected['TRAIN']
    proc=codec.processor(read(statpath));gate=TrajectorySafetyGateV1();results=[]
    for split in ('TRAIN','KIN-VAL'):
        for family in selected[split]:
            sequence_results=[]
            for seq in [s for s in reg['sequences'] if s['family_id']==family]:
                output=ROOT/'dataset'/seq['sequence_id'];receipt_path=output.with_suffix('.json')
                if receipt_path.exists():
                    r=read(receipt_path);assert r['statistics_sha256']==sha(statpath) and r['codec_sha256']==sha(codec.__file__) and sha(output.with_suffix('.npz'))==r['array_sha256']
                    sequence_results.append(r);continue
                rows=[];states=[];labels=[];encoded=[];normalized_states=[];masks=[];decoded=[]
                for row,state,label in windows(reg,seq):
                    ps,pa,mask=codec.encode(proc,state,label);roundtrip=codec.decode(proc,pa,state)
                    rotations=rot6d_rows_to_matrix(roundtrip[:,3:9]);target_rotations=rot6d_rows_to_matrix(label[:,3:9])
                    pe=float(np.linalg.norm(roundtrip[:,:3]-label[:,:3],axis=1).max())
                    re=float(Rotation.from_matrix(rotations@target_rotations.transpose(0,2,1)).magnitude().max())
                    finite=bool(np.isfinite(roundtrip).all() and np.isfinite(pa).all() and np.isfinite(ps).all())
                    mask_ok=bool(np.all(mask[:,:10]==1) and np.all(mask[:,10:]==0) and np.all(pa[:,10:]==0) and np.all(ps[:,10:]==0))
                    t=row['time'];chunk=TrajectoryChunkV1(seq['sequence_id']+'-'+str(row['index']),codec.IDENTITY,t,t+np.arange(40)*.05,
                        roundtrip[:,:3],Rotation.from_matrix(rotations).as_quat(),'NATIVE_WBC_RELATIVE_REFERENCE','x5_link6_gripper_center',t,t+1.95,gripper=roundtrip[:,9])
                    verdict=gate.evaluate(chunk,now=t,achieved_position=state[:3],achieved_orientation=Rotation.from_matrix(rot6d_rows_to_matrix(state[3:9])).as_quat()).as_dict()
                    valid=finite and mask_ok and pe<=1e-5 and re<=1e-4 and verdict['decision']!='REJECT'
                    row.update(codec_valid=valid,position_roundtrip_m=pe,rotation_roundtrip_rad=re,mask_exact=mask_ok,codec_gate=verdict,
                        usable=row['raw_good'] and row['visible'] and valid)
                    rows.append(row);states.append(state);labels.append(label);encoded.append(pa);normalized_states.append(ps);masks.append(mask);decoded.append(roundtrip)
                active=[r for r in rows if r['active']];end=reg['experts'][seq['task']]['motion_end_s']
                fraction=sum(r['usable'] for r in active)/len(active)
                quarters=[any(r['usable'] and k*end/4<=r['time']<(k+1)*end/4 for r in active) for k in range(4)]
                required=[r for r in rows if any(abs(r['time']-t)<1e-8 for t in reg['renderer']['calibration_times'][seq['task']])]
                qualified=fraction>=.95 and all(quarters) and all(r['usable'] for r in required)
                output.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output.with_suffix('.npz'),state_v1=states,raw_action=labels,action=encoded,state=normalized_states,
                    action_mask=masks,decoded_action=decoded,time=[r['time'] for r in rows],indices=[r['index'] for r in rows])
                r=dict(sequence=seq,rows=rows,qualified=qualified,usable_fraction=fraction,quartiles=quarters,
                    required_pass=all(r['usable'] for r in required),statistics_sha256=sha(statpath),codec_sha256=sha(codec.__file__),
                    array_sha256=sha(output.with_suffix('.npz')),max_position_error=max(x['position_roundtrip_m'] for x in rows),
                    max_rotation_error=max(x['rotation_roundtrip_rad'] for x in rows))
                write(receipt_path,r);sequence_results.append(r)
            results.append(dict(family=family,split=split,qualified=all(s['qualified'] for s in sequence_results),
                sequences=[dict(sequence=s['sequence']['sequence_id'],qualified=s['qualified'],fraction=s['usable_fraction'],
                    max_position_error=s['max_position_error'],max_rotation_error=s['max_rotation_error']) for s in sequence_results]))
            print(json.dumps(dict(family=family,split=split,qualified=results[-1]['qualified'])),flush=True)
    final={s:post_codec(selected[s],{r['family']:r['qualified'] for r in results if r['split']==s}) for s in selected}
    result=tier(len(final['TRAIN']),len(final['KIN-VAL']))
    write(ROOT/'dataset/codec_receipt.json',dict(status=result,selected_pre_codec=selected,selected_post_codec=final,families=results,
        statistics_sha256=sha(statpath),statistics_fit_count=1,codec_sha256=sha(codec.__file__),geometry_receipt_sha256=sha(ROOT/'review_geometry/receipt.json'),
        H1_status='CODEC_COMPLETE_INPUT_EVALUATION_READINESS_PENDING',new_model_queries=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status=result,train=len(final['TRAIN']),val=len(final['KIN-VAL']))),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='seal':seal(sys.argv[2])
    elif sys.argv[1]=='codec':build()
