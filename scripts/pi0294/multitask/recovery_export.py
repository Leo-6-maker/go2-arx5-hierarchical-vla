"""Expert-only recovery suffix beside five immutable book TRAIN episodes.

Use the existing committed-command packet, official reader and codec. The
mixed episode's student prefix is discarded before fitting or serialization.
No engine, model, new RGB, optimizer, statistics overwrite or DEV selection.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
from scripts.pi0292.train_adapter import DOMAIN,TAG,OFFICIAL_SHA,sha
from .source_export import packet,statistics,KINDS
from .book_recovery import CONDITION,TAKEOVER,CHECKPOINT,recovery_card,eligible_recovery_anchors
from .libero_book_teacher import teacher_for_card,route_card,RELEASE_REVISION

PARENT_DATA='47163b091f60c6d30e49b50699177727f8909e2100533fa4bc4555b8431c010d'
PARENT_STATS='060ff05d4c34205593e234d43a5a76bd5c2305171289489d51fb78b68122e096'
SEAL='3d5c4a9ad1a94b6c68f38644e1b9a6dc67f7735806667851049ac250b9cae1cf'
RECOVERY_ID='BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX'
COUNTS=(601,601,601,601,601,569)


def raw(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()


def expert_suffix_packet(full):
    if (full['actual_state10'].shape!=(641,10)or full['committed_absolute10'].shape!=(640,10)
            or not np.array_equal(full['valid_window'],np.arange(640)<601)):
        raise ValueError('FIRST_VALIDATE_THE_COMPLETE_ACTUAL_SOURCE_PACKET')
    suffix={k:full[k][TAKEOVER:].copy()for k in('actual_state10','committed_absolute10','rgb_paths','rgb_sha256')}
    suffix.update(valid_window=np.arange(608)<569,anchor_time_s=np.arange(TAKEOVER,601)/20,
        source_observation_index=np.arange(TAKEOVER,641),offset_s=full['offset_s'].copy(),
        instruction=full['instruction'].copy(),label_representation=full['label_representation'].copy())
    assert suffix['actual_state10'].shape==(609,10)and suffix['committed_absolute10'].shape==(608,10)
    return suffix


def fit_statistics(packets):
    states=[];actions=[];relative=[]
    if len(packets)!=6:raise ValueError('EXACT_FIVE_FULL_TRAIN_PLUS_ONE_EXPERT_SUFFIX')
    for p,count in zip(packets,COUNTS):
        c=p['committed_absolute10'];s=p['actual_state10'][:count]
        if c.shape!=(count+39,10)or s.shape!=(count,10)or not np.all(p['valid_window'][:count]):
            raise ValueError('ALL_AND_ONLY_ACTUAL_COMPLETE_EXPERT_H40_WINDOWS')
        future=np.arange(count)[:,None]+np.arange(40)
        a=c[future];states.append(s);actions.append(a)
        relative.append(pose10(np.linalg.inv(matrix_from10(s))[:,None]@matrix_from10(c)[future],a[...,9])[...,:9])
    s,a,z=np.concatenate(states),np.concatenate(actions),np.concatenate(relative)
    return {TAG:dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:],(0,1))),
        relative_action=dict(eef_9d=statistics(z,(0,1))))}


def prepare(parent,folder,output,*,pins):
    parent,folder,output=Path(parent),Path(folder),Path(output)
    if output.exists():raise FileExistsError('PRESERVE_ORIGINAL_AND_PRIOR_DERIVATIVES')
    if (sha(parent/'DATASET_MANIFEST.json')!=PARENT_DATA or sha(parent/'statistics.json')!=PARENT_STATS
            or sha(parent/'TRAIN_DEV_SEAL.json')!=SEAL):
        raise ValueError('EXACT_EXISTING_BOOK5_DATA_STATS_AND_UNCHANGED_DEV')
    old=json.loads((parent/'DATASET_MANIFEST.json').read_bytes())
    if (old['training_admitted']is not True or len(old['episodes'])!=5
            or [r['id']for r in old['episodes']]!=[f'BOOK_SOURCE_SEED{i:03}'for i in range(5)]):
        raise PermissionError('ONLY_EXISTING_FIVE_ADMITTED_TRAIN_EPISODES')
    run=folder.parent.parent
    prefix='conditions/BOOK_RECOVERY_TRAIN000/'
    required={'RUN_MANIFEST.json','control/LEASE.json','RESULT.json',prefix+'CARD.json',
        prefix+'INDEPENDENT_RAW_AUDIT.json',prefix+'TERMINAL.json',prefix+'predictions/04_expert_takeover.json'}
    if set(pins)!=required or folder!=run/'conditions/BOOK_RECOVERY_TRAIN000':
        raise ValueError('EXACT_CLOSED_RECOVERY_PRODUCER_PIN_SET')
    for name,pin in pins.items():
        if name.startswith('/')or'..'in Path(name).parts or sha(run/name)!=pin:
            raise ValueError('ACTUAL_CLOSED_RECOVERY_PRODUCER_CHANGED:'+name)
    lease=json.loads((run/'control/LEASE.json').read_bytes());result=json.loads((run/'RESULT.json').read_bytes())
    producer=json.loads((run/'RUN_MANIFEST.json').read_bytes())
    audit=json.loads((folder/'INDEPENDENT_RAW_AUDIT.json').read_bytes());card=json.loads((folder/'CARD.json').read_bytes())
    if (lease['status']!='CLOSED'or lease['manifest_sha256']!=sha(run/'RUN_MANIFEST.json')
            or lease['result_sha256']!=sha(run/'RESULT.json')
            or producer['kind']!='V7_BOOK_TRAIN_RECOVERY_DIAGNOSTIC'
            or producer['code_source_sha']!='dde03055f0a379bf64ffe68549417ac914f989be'
            or result['evaluated_checkpoint_manifest_sha256']!=CHECKPOINT
            or result['status']!='COMPLETE_TRAIN_RECOVERY_DIAGNOSTIC'
            or result['formal_DEV_conditions']!=0 or result['training_data_admission']is not False
            or result['learned_full_task_success_claim']is not False or result['model_queries']!=4
            or not audit['complete_episode']or not audit['task_success']or not audit['original_native_safety_pass']
            or audit['integrity_problems']or card!=recovery_card(CONDITION)):
        raise PermissionError('ONE_CLOSED_FULL_SAFE_EXISTING_TRAIN_RECOVERY_NO_DEV')
    if sha(folder/'ACTUAL_RECORDS.jsonl')!=audit['raw_sha256']:raise ValueError('FULL_RAW_SOURCE_CHANGED')
    rows=[]
    with(folder/'ACTUAL_RECORDS.jsonl').open('rb')as f:
        for line in f:
            item=json.loads(line)
            if item['kind']in KINDS:rows.append(item)
    full=packet(rows,card);obs={r['value']['index']:r['value']for r in rows if r['kind']=='observation'}
    done=[r['value']['observation_index']for r in rows if r['kind']=='completed_interval']
    anchors=eligible_recovery_anchors(done,full_task_success=audit['task_success'],native_safe=audit['original_native_safety_pass'])
    if anchors!=tuple(range(32,601)):raise ValueError('EXACT_COMPLETED_EXPERT_SUFFIX')
    takeover=json.loads((folder/'predictions/04_expert_takeover.json').read_bytes())
    if takeover['index']!=32 or takeover['student_prefix_is_expert']is not False:
        raise ValueError('EXPLICIT_ACTUAL_EXPERT_TAKEOVER_BOUNDARY')
    np.testing.assert_array_equal(takeover['actual_state_eef10'],full['actual_state10'][32])
    plans={r['value']['observation_index']:r['value']for r in rows if r['kind']=='planned_H40'}
    teacher=teacher_for_card(route_card(RELEASE_REVISION));replayed=0
    for i in range(32,641):
        teacher.observe(obs[i]['actual_state'])
        if i<640 and i%8==0:
            p=teacher.plan();expected=np.stack([pose10(T,g)for T,g in zip(p['poses'],p['opening'])])
            np.testing.assert_allclose(expected,plans[i]['rows10'],rtol=0,atol=1e-10);replayed+=1
    if replayed!=76 or not teacher.candidate_complete:raise ValueError('ACTUAL_EXPERT_SUFFIX_REPLAY_OR_HOLD')
    from PIL import Image
    for i in range(32,641):
        for name in('body_rgb','wrist_rgb'):
            pin=obs[i]['policy_obs'][name];image=folder/pin['file']
            if image.is_symlink()or image.stat().st_size!=pin['bytes']or sha(image)!=pin['sha256']:
                raise ValueError('ACTUAL_RECOVERY_RGB_CHANGED')
            with Image.open(image)as im:
                if im.mode!='RGB'or im.size!=(256,256):raise ValueError('ACTUAL_256_RGB')
                im.load()
    suffix=expert_suffix_packet(full);rows=[];packets=[]
    for row in old['episodes']:
        if row['split']!='TRAIN'or row['windows']!=601 or sha(parent/row['file'])!=row['sha256']:
            raise ValueError('ORIGINAL_BOOK5_ROW_OR_NUMERIC_CHANGED')
        with np.load(parent/row['file'],allow_pickle=False)as a:packets.append({k:a[k].copy()for k in a.files})
        rows.append(dict(row))
    packets.append(suffix)
    output.mkdir(parents=True,exist_ok=False);(output/'episodes').mkdir()
    for row in rows:
        shutil.copyfile(parent/row['file'],output/row['file'])
        if sha(output/row['file'])!=row['sha256']:raise ValueError('ORIGINAL_NUMERIC_COPY_READBACK')
    numeric=output/'episodes'/f'{RECOVERY_ID}.npz';np.savez_compressed(numeric,**suffix)
    rows.append(dict(id=RECOVERY_ID,task='BOOK',family='LIBERO_ARX5_LEFT_PORT',origin='SOURCE_EXPERT_RECOVERY_SUFFIX_NOT_FULL_TEACHER',
        split='TRAIN',windows=569,file=numeric.relative_to(output).as_posix(),sha256=sha(numeric),raw_root=str(folder),
        instruction=card['instruction'],raw_sha256=audit['raw_sha256'],audit_sha256=sha(folder/'INDEPENDENT_RAW_AUDIT.json'),
        original_TRAIN_parent='BOOK_SOURCE_SEED000',source_observation_start=32,source_observation_end=640,
        complete_commands=608,actual_observations=609,student_prefix_expert_labels=0,source_full_task_success=True,
        full_pure_Teacher_episode=False,independent_new_layout=False))
    def put(name,value):
        with(output/name).open('xb')as f:f.write(raw(value))
    put('statistics.json',fit_statistics(packets));shutil.copyfile(parent/'TRAIN_DEV_SEAL.json',output/'TRAIN_DEV_SEAL.json')
    put('STATISTICS_PROVENANCE.json',dict(rule='FIVE_TRAIN_FULL_EPISODES_PLUS_ONE_ACTUAL_EXPERT_SUFFIX_EQUAL_WINDOW_TIMESTEP_MINMAX_V1',
        source_parent_data_manifest_sha256=PARENT_DATA,source_parent_statistics_sha256=PARENT_STATS,
        state_anchor_rows=3574,action_window_timestep_rows=142960,student_prefix_rows=0,DEV_or_TEST_rows=0,
        old_E1_E2_R0_or_original_book5_overwritten=False,full_Teacher_episodes=5,mixed_recovery_episodes=1))
    m=dict(schema='BOOK5_PLUS_ONE_EXPERT_RECOVERY_SUFFIX_V1',domain=DOMAIN,purpose='NEW_BOOK_RECOVERY_LEARNING_SEPARATE_VERSION',
        training_admitted=False,official_source_sha=OFFICIAL_SHA,episodes=rows,statistics_sha256=sha(output/'statistics.json'),
        TRAIN_DEV_seal_sha256=sha(output/'TRAIN_DEV_SEAL.json'),TRAIN_windows=3574,DEV_windows=0,DEV_conditions=10,
        qualified_pure_Teacher_episodes=5,mixed_student_expert_recovery_episodes=1,
        distinct_initial_conditions=5,scene_geometry_families=1,
        student_prefix_expert_labels=0,actual_expert_H40s_replayed=76,all_609_recovery_RGB_pairs_verified=True,
        original_book5_manifest_sha256=PARENT_DATA,new_models_queries_optimizer_physics=0)
    put('DATASET_MANIFEST.json',m);return m,packets


def admit(root):
    root=Path(root);m=json.loads((root/'DATASET_MANIFEST.json').read_bytes());q=json.loads((root/'OFFICIAL_QUALIFICATION.json').read_bytes())
    if (m['schema']!='BOOK5_PLUS_ONE_EXPERT_RECOVERY_SUFFIX_V1'or m['training_admitted']is not False
            or q['candidate_manifest_sha256']!=sha(root/'DATASET_MANIFEST.json')
            or q['all_3574_loader_anchors']is not True or q['all_609_recovery_RGB_pairs_verified']is not True
            or q['CUDA_initialized']is not False or q['new_models_queries_optimizer_physics_PBS']!=0
            or len(q['codecs'])!=6 or [x['H40_windows_actually_checked']for x in q['codecs']]!=list(COUNTS)
            or any(x['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'or x['native_inverse_clip_components_float32']!=0 for x in q['codecs'])):
        raise PermissionError('ACTUAL_ORIGINAL_OFFICIAL_CODEC_AND_EXPERT_ONLY_QUALIFICATION_REQUIRED')
    admitted=root/'admitted';admitted.mkdir(exist_ok=False);(admitted/'episodes').mkdir()
    for row in m['episodes']:
        shutil.copyfile(root/row['file'],admitted/row['file'])
        if sha(admitted/row['file'])!=row['sha256']:raise ValueError('ADMITTED_NUMERIC_READBACK')
    for name in('statistics.json','TRAIN_DEV_SEAL.json','STATISTICS_PROVENANCE.json'):shutil.copyfile(root/name,admitted/name)
    m.update(training_admitted=True,official_qualification_sha256=sha(root/'OFFICIAL_QUALIFICATION.json'),
        original_candidate_manifest_sha256=sha(root/'DATASET_MANIFEST.json'),original_raw_flags_unchanged=True)
    with(admitted/'DATASET_MANIFEST.json').open('xb')as f:f.write(raw(m))
    return admitted,m
