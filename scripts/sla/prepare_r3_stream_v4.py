"""Deterministic recorded RGB stream bank; no model/native/label refitting."""
from pathlib import Path
import json
import numpy as np
from scripts.pi0292.durable import write
from scripts.sla.export_mfree_c20_v1 import _reference
from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width
from go2arx5_vla.world_model.storage import digest

RULE='FIRST_SORTED_QUALIFIED_EXTRA_EP_PER_FAMILY_FIRST_CLOSE_MINUS6_CONTIGUOUS16_NOISE0'
def read(p):return json.loads(Path(p).read_bytes())


def choose(rows):
    first=next(i for i,r in enumerate(rows)if r['phase']=='CLOSE')
    start=max(0,first-6)
    selected=list(range(start,start+16))
    if selected[-1]>=len(rows):raise ValueError('NO_FULL_PREDECLARED_CONTIGUOUS16_SEGMENT')
    for a,b in zip(selected,selected[1:]):
        if (rows[b]['native_step']-rows[a]['native_step']!=250
                or abs(rows[b]['native_time_s']-rows[a]['native_time_s']-.5)>1e-8):
            raise ValueError('ORIGINAL_CONTIGUOUS_500MS_CLOCK_REQUIRED')
    return selected


def build(spec, destination):
    source=spec['CN38_bank']
    if digest(source['path'])!=source['sha256']:raise ValueError('SEALED_CN38_BANK_CHANGED')
    bank=read(source['path']);root=Path(destination);root.mkdir();records=[];episodes=[]
    for family in ('F1','F2'):
        ep=sorted((e for e in bank['episodes']if e['family']==family),key=lambda e:e['id'])[0]
        for key in ('result','spec','capture','labels'):
            if digest(ep[key]['path'])!=ep[key]['sha256']:raise ValueError('ORIGINAL_QUALIFIED_EPISODE_CHANGED')
        capture=read(ep['capture']['path']);rows=read(ep['labels']['path']);native=read(ep['spec']['path'])
        sample=Path(ep['result']['path']).parent/'native-01/MOBILE_SAMPLES.private.jsonl'
        if digest(sample)!=ep['command_samples_sha256']:raise ValueError('RECORDED_COMMAND_HISTORY_CHANGED')
        previous={s['native_end_step']:s for s in map(json.loads,sample.read_text().splitlines())}
        selected=choose(rows);cache={};frames=[]
        for anchor in selected:
            row=rows[anchor];descriptor=next(d for d in capture['frame_shards']if d['first_query']<=anchor<d['first_query']+d['count'])
            file=descriptor['path']
            if file not in cache:
                actual_sha=digest(file)
                if actual_sha!=descriptor['sha256']:raise ValueError('ORIGINAL_RGB_SHARD_BYTES_CHANGED')
                with np.load(file,allow_pickle=False)as z:cache[file]={k:z[k].copy()for k in ('rgb','native_step','state_eef10')}
                frames.append(dict(path=file,sha256=actual_sha))
            z=cache[file];j=anchor-descriptor['first_query'];state=np.asarray(row['state_eef10']);p=previous[row['native_step']]
            if int(z['native_step'][j])!=row['native_step'] or not np.array_equal(state,z['state_eef10'][j]):raise ValueError('SAME_ACTUAL_RECORDED_RGB_STATE_CLOCK')
            reference=dict(width_m=.044*float(p['installed_EEF10'][9]),source_native_s=p['gripper_command']['source_native_s'],query_native_s=row['native_time_s'])
            reference_width(reference)
            f=root/('%02d-INPUT.npz'%len(records))
            with f.open('xb')as handle:np.savez(handle,state_eef10=state,body_rgb=z['rgb'][j][0],wrist_rgb=z['rgb'][j][1],
                T_policy_from_mount=np.asarray(row['episode_fixed_T_policy_from_mount']),policy_from_world=np.asarray(native['policy_from_world']),accepted_reference_TCP=_reference(row))
            records.append(dict(index=len(records),family=family,episode=ep['id'],anchor=anchor,phase_score_only=row['phase'],noise=0,
                native_step=row['native_step'],recorded_capture_s=row['native_time_s'],previous_reference=reference,
                input=dict(path=str(f),sha256=digest(f)),language=native['language'],recorded_previous_command_not_student_feedback=True))
        episodes.append(dict(id=ep['id'],anchors=selected,original=ep,frame_shards=frames))
    result=dict(schema='R3_RECORDED_CONTIGUOUS16_PER_FAMILY_V4',selection_rule=RULE,records=records,episodes=episodes,
        no_future_RGB_prefetch_in_measured_controller=True,offline_bank_preparation_is_not_camera_delivery=True,
        new_model_generations=0,new_native_requests=0,new_integrations=0)
    write(root/'MANIFEST.private.json',result);return dict(path=str(root/'MANIFEST.private.json'),sha256=digest(root/'MANIFEST.private.json'))
