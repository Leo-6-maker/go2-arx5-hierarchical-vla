"""Seal fixed episode-heldout inputs; never fit statistics or create training data."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.pi0292.durable import write
from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
from scripts.sla.export_mfree_c20_v1 import _reference
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width

PHASES=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')
read=lambda p:json.loads(Path(p).read_bytes())


def build(spec,destination):
    from scripts.wm02.mfree_collection_v1 import capture_check
    root=Path(destination);root.mkdir(exist_ok=False)
    snapshot=spec['CN38_snapshot'];c20=Path(spec['dataset'])
    if sha(snapshot['path'])!=snapshot['sha256']or sha(c20/'MANIFEST.private.json')!=spec['dense_manifest_sha256']:
        raise ValueError('PINNED_CN38_AND_C20_MEMBERSHIP_REQUIRED')
    population=read(snapshot['path']);training=read(c20/'MANIFEST.private.json')['episodes']
    old={e['id']for e in training};old_results={e['native_result']['sha256']for e in training}
    extra=sorted((r for r in population['episodes']if r['id']not in old),key=lambda r:r['id'])
    if len(extra)!=18 or Counter(r['family']for r in extra)!=Counter(F1=14,F2=4):
        raise ValueError('EXACT_CN38_MINUS_C20_F114_F24_REQUIRED')
    records=[];episode_pins=[];checked_models={}
    for episode in extra:
        if episode['state']!='QUALIFIED':raise ValueError('ONLY_ORIGINAL_QUALIFIED_EPISODES')
        if episode['result']['sha256']in old_results:raise ValueError('ALIASED_C20_RESULT_IS_NOT_EPISODE_HELDOUT')
        for key in ('result','spec'):
            if sha(episode[key]['path'])!=episode[key]['sha256']:raise ValueError('ORIGINAL_EPISODE_IDENTITY_CHANGED')
        native=read(episode['result']['path']);ns=read(episode['spec']['path']);capture_check(native)
        capture=read(native['detail']['capture']['path']);rows=read(capture['labels']['path'])
        model=ns['model_path']
        if model not in checked_models:checked_models[model]=sha(model)
        if checked_models[model]!=capture['frozen_scene_sha256']:raise ValueError('ACTUAL_FROZEN_SCENE_IDENTITY')
        selected=[]
        for phase in PHASES:
            anchors=[i for i,r in enumerate(rows)if r['phase']==phase]
            if not anchors:raise ValueError('ALL_FIVE_PHASES_REQUIRED')
            selected.append((phase,anchors[(len(anchors)-1)//2]))
        sample_file=Path(episode['result']['path']).parent/'native-01/MOBILE_SAMPLES.private.jsonl'
        previous={r['native_end_step']:r for r in map(json.loads,sample_file.read_text().splitlines())}
        cache={};folder=root/episode['id'];folder.mkdir()
        for phase,anchor in selected:
            row=rows[anchor];descriptor=next(s for s in capture['frame_shards']if s['first_query']<=anchor<s['first_query']+s['count'])
            file=descriptor['path']
            if file not in cache:
                with np.load(file,allow_pickle=False)as z:cache[file]={k:z[k].copy()for k in ('rgb','native_step','state_eef10')}
            z=cache[file];j=anchor-descriptor['first_query'];state=np.asarray(row['state_eef10'])
            if (int(z['native_step'][j])!=row['native_step']or not np.array_equal(state,z['state_eef10'][j])
                    or z['rgb'][j].shape!=(2,256,256,3)):
                raise ValueError('EXACT_CAPTURED_CURRENT_RGB_STATE_CLOCK')
            prev=previous[row['native_step']];context=dict(width_m=.044*float(prev['installed_EEF10'][9]),
                source_native_s=prev['gripper_command']['source_native_s'],query_native_s=row['native_time_s'])
            reference_width(context);transform=np.asarray(ns['policy_from_world']);world=np.asarray(row['nominal_world_H40'])
            expert=pose10(transform@matrix_from10(world),world[:,9]);frame=np.asarray(row['episode_fixed_T_policy_from_mount'])
            inp=folder/(phase+'-INPUT.npz');truth=folder/(phase+'-EXPERT.npz')
            with inp.open('xb')as f:np.savez(f,state_eef10=state,body_rgb=z['rgb'][j][0],wrist_rgb=z['rgb'][j][1],T_policy_from_mount=frame)
            with truth.open('xb')as f:np.savez(f,physical_H40=expert,accepted_reference_TCP=_reference(row))
            records.append(dict(index=len(records),episode=episode['id'],family=episode['family'],phase=phase,anchor=anchor,
                native_step=row['native_step'],language=ns['language'],previous_reference=context,
                policy_from_world=ns['policy_from_world'],input=dict(path=str(inp),sha256=sha(inp)),
                truth=dict(path=str(truth),sha256=sha(truth)),actual_scene_model_sha256=checked_models[model],
                raw_scene_lineage_diagnostic_only=ns['collection_contract']['scene_lineage']))
        episode_pins.append(dict(id=episode['id'],family=episode['family'],spec=episode['spec'],result=episode['result'],
            capture=native['detail']['capture'],labels=capture['labels'],command_samples_sha256=sha(sample_file),
            actual_model_sha256=checked_models[model]))
    if len(records)!=90:raise ValueError('EXACT18_TIMES5_ANCHORS')
    result=dict(schema='CN38_MINUS_C20_HYBRID_OFFLINE_BANK_V1',records=records,episodes=episode_pins,
        anchor_rule='LOWER_MEDIAN_QUERY_INDEX_WITHIN_EACH_ORIGINAL_FIVE_PHASES',noise_ids=[0,1],
        conditions=180,new_model_generation_cap=360,split='EPISODE_HELDOUT_DIAGNOSTIC_NOT_BLIND_TEST',
        source_snapshot=snapshot,excluded_C20_manifest_sha256=spec['dense_manifest_sha256'],statistics_refitted=False,
        new_model_calls=0,new_updates=0,new_native_requests=0,new_integrations=0,object_truth_in_model_input=False)
    write(root/'MANIFEST.private.json',result);return dict(path=str(root/'MANIFEST.private.json'),sha256=sha(root/'MANIFEST.private.json'))
