"""Independent repeat CPU score and raw-manifest assembly after all pilots."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from go2arx5_vla.task_data.config import COMMISSIONING,PILOTS,PARAMETERS as P,canonical_hash
from go2arx5_vla.task_data.engine import POINTS
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget


def rescore(folder):
    card=json.loads((folder/'CARD.json').read_text())
    with np.load(folder/'physics.npz',allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    t=d['time_s'];xyz=d['object_pose'][:,:3,3]
    grasp=np.logical_and.reduce([d[k] for k in ['object_present','contact_valid','gripper_state_valid','native_grasp','left_object_contact','right_object_contact']])
    def sustained(mask):
        starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
        ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
        return bool(any(t[b]-t[a]>=.3-1e-8 and np.all(np.diff(t[a:b+1])<=.05+1e-8) for a,b in zip(starts,ends)))
    history=dict(sustained_grasp=sustained(grasp),sustained_lift=sustained(grasp&(xyz[:,2]-xyz[0,2]>=.02)),
        horizontal_carry=bool((grasp&(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1)>=.10)).any()))
    corners=np.asarray([pose[:3,3]+POINTS@pose[:3,:3].T for pose in d['object_pose']])
    checks=[(np.abs(corners[:,:,:2]-card['goal_xy'])<=.07).all(axis=(1,2)),
        np.abs(xyz[:,2]-.42)<=.005,np.linalg.norm(d['object_linear_velocity'],axis=1)<=.02,
        np.linalg.norm(d['object_angular_velocity'],axis=1)<=.10,~grasp & ~d['native_grasp'],
        np.linalg.norm(d['tcp_world'][:,:3,3]-xyz,axis=1)>=.08,
        d['object_present']&d['contact_valid']&d['gripper_state_valid']]
    complete=len(t)==641 and np.all(np.abs(t-np.arange(641)/20)<=1e-8)
    safe=bool(d['safety_valid'].all() and not any(d[k].any() for k in ['native_abort','illegal_contact','joint_limit_penetration','reset','takeover']))
    high=np.load(folder/'highrate.npz')['rows'];safe=bool(safe and not high[:,-1].any())
    hold=int(np.logical_and.reduce(checks)[560:601].sum()) if complete else 0
    result=dict(history=history,hold_pass=hold,persistent_success=bool(complete and safe and all(history.values()) and hold==41),
        complete=bool(complete),highrate_safety_valid=safe,physics_sha256=file_hash(folder/'physics.npz'))
    stored=json.loads((folder/'SCORE.json').read_text())
    for k in ['history','hold_pass','persistent_success']:assert result[k]==stored[k],(folder.name,k)
    return result


p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit-source-sha',required=True);a=p.parse_args()
root=Path(a.root);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
try:
    audit=json.loads((root/'evidence/DATA_ADMISSION_AND_CODEC.json').read_text());assert audit['status']=='PASS_STATIC_DATA_AND_NATIVE_CODEC_ONLY'
    pilots=[];raw=[]
    for identifier in COMMISSIONING+PILOTS:
        folder=root/'episodes'/identifier;result=json.loads((folder/'RESULT.json').read_text());files=json.loads((folder/'FILE_MANIFEST.json').read_text())
        record=dict(id=identifier,result=result,file_manifest_sha256=file_hash(folder/'FILE_MANIFEST.json'),
            file_count=len(files),bytes=sum(x['bytes'] for x in files),paired_frames=result['paired_samples'],
            data_location='Corelab PI0290/episodes/'+identifier,streams=['policy_obs','causal_history','supervision','execution','privileged','geometry'],
            original_failures_preserved=True,quality_retries=0)
        raw.append(record)
        if identifier in PILOTS:
            first=rescore(folder);second=rescore(folder);assert first==second
            pilots.append(dict(id=identifier,split='HELDOUT' if identifier in PILOTS[4:] else 'SEEN',
                score=first,repeat_score_sha256=canonical_hash(first),independent_repeat_match=True))
    passed=sum(x['score']['persistent_success'] for x in pilots);held=sum(x['score']['persistent_success'] for x in pilots[4:])
    all_complete=all(x['score']['complete'] for x in pilots)
    ready=all_complete and passed>=7 and held>=3
    score=dict(status='INDEPENDENT_DATA_PILOT_READY' if ready else 'PILOT_VALID_QUALITY_FAIL',
        source_sha=json.loads((root/'TEACHER_AND_LAYOUT_SEAL.json').read_text())['source_sha'],audit_source_sha=a.audit_source_sha,
        first_attempt_denominator=8,complete_count=sum(x['score']['complete'] for x in pilots),successes=passed,heldout_successes=held,
        pilots=pilots,thresholds_changed=False,selected='NONE',deployment='NONE',domain='INDEPENDENT_DYNAMIC_TOOL_SIM_ONLY_NON_REALTIME')
    write(root/'evidence/PILOT_SCORE.json',score)
    write(root/'evidence/PILOT_RAW_MANIFEST.json',dict(schema='INDEPENDENT_DYNAMIC_TOOL_DATA_V1',episodes=raw,
        commissioning_only_for_qualification=True,dataset_admitted_ids=list(PILOTS),
        commissioning_visual_caveat='original wrist occlusion; no training admission; diagnostics never substitute original observations',
        all_attempts_included=True,source_sha=score['source_sha']))
    print(json.dumps(score))
finally:budget.finish_cpu('D3_INDEPENDENT_REPEAT_SCORE_AND_MANIFEST')
