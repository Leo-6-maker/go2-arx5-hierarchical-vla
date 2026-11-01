"""Reuse the original packet/codec path; emit only executed late expert suffixes.

One numeric candidate and provenance only. Batch-level TRAIN statistics and
official qualification/admission occur after the new population is frozen.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.train_adapter import sha
from .source_export import packet,KINDS
from . import book_late_recovery as recovery
from .book_held_teacher import from_held_state


def suffix_packet(full,takeover):
    if(type(takeover)is not int or takeover%8 or not 8<=takeover<=480
            or full['actual_state10'].shape!=(641,10)or full['committed_absolute10'].shape!=(640,10)
            or not np.array_equal(full['valid_window'],np.arange(640)<601)):
        raise ValueError('COMPLETE_EXECUTED_PACKET_AND_REAL_LATE_SOURCE_BOUNDARY')
    result={k:full[k][takeover:].copy()for k in('actual_state10','committed_absolute10','rgb_paths','rgb_sha256')}
    result.update(valid_window=np.arange(640-takeover)<601-takeover,
        source_observation_index=np.arange(takeover,641),anchor_time_s=np.arange(takeover,601)/20,
        offset_s=full['offset_s'].copy(),instruction=full['instruction'].copy(),
        label_representation=full['label_representation'].copy())
    return result


def export_one(folder,output,*,audit_sha256,terminal_sha256,raw_sha256):
    folder=Path(folder);output=Path(output);run=folder.parent.parent
    read=lambda p:json.loads(Path(p).read_bytes())
    if output.exists():raise FileExistsError('PRESERVE_PRIOR_RAW_AND_NUMERIC_CANDIDATES')
    for name,pin in(('INDEPENDENT_RAW_AUDIT.json',audit_sha256),('TERMINAL.json',terminal_sha256),('ACTUAL_RECORDS.jsonl',raw_sha256)):
        if sha(folder/name)!=pin:raise PermissionError('CLOSED_ACTUAL_LATE_RECOVERY_MEMBER_CHANGED')
    m=read(run/'RUN_MANIFEST.json');recovery.validate_manifest(m)
    lease=read(run/'control/LEASE.json');result=read(run/'RESULT.json');audit=read(folder/'INDEPENDENT_RAW_AUDIT.json');card=read(folder/'CARD.json')
    condition=next((c for c in m['recovery_plan']['conditions']if c['id']==folder.name),None)
    if(condition is None or card!=recovery.recovery_card(condition)or lease['status']!='CLOSED'
            or lease['manifest_sha256']!=sha(run/'RUN_MANIFEST.json')or lease['result_sha256']!=sha(run/'RESULT.json')
            or result['formal_DEV_conditions']!=0 or result['training_data_admission']is not False
            or not audit['complete_episode']or not audit['task_success']or not audit['original_native_safety_pass']or audit['integrity_problems']):
        raise PermissionError('FULL_SAFE_CLOSED_LATE_TRAIN_RECOVERY_NOT_DEV_REQUIRED')
    case=next(c for c in result['conditions']if c['id']==condition['id'])
    if not case['qualified_expert_recovery']:raise PermissionError('NO_EVENT_OR_FAILED_RECOVERY_HAS_NO_EXPERT_LABELS')
    takeover=case['late_recovery']['takeover_index'];marker=folder/'predictions'/f'{takeover//8:02}_expert_takeover.json'
    hand=read(marker);snapshot=hand['full_state']['snapshot']
    encoded=json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if(hashlib.sha256(encoded).hexdigest()!=hand['full_state']['sha256']or hand['index']!=takeover
            or hand['condition']!=condition or hand['student_prefix_is_expert']is not False
            or hand['full_state_unchanged_by_initialization']is not True):
        raise PermissionError('ACTUAL_UNCHANGED_FULL_STATE_HELD_HANDOVER_REQUIRED')
    rows=[]
    with(folder/'ACTUAL_RECORDS.jsonl').open()as f:
        for line in f:
            r=json.loads(line)
            if r['kind']in KINDS:rows.append(r)
    full=packet(rows,card);obs={r['value']['index']:r['value']for r in rows if r['kind']=='observation'}
    done=[r['value']['observation_index']for r in rows if r['kind']=='completed_interval']
    anchors=recovery.eligible_recovery_anchors(condition,takeover,done,full_task_success=True,native_safe=True)
    np.testing.assert_array_equal(hand['actual_state_eef10'],full['actual_state10'][takeover])
    for key in('qpos','qvel','tcp','object_pose','finger_q','finger_qd'):
        np.testing.assert_array_equal(snapshot['actual_state'][key],obs[takeover]['actual_state'][key])
    teacher=from_held_state(recovery.original_card(condition),obs[takeover]['actual_state'],source_index=takeover,
        grasp_since=hand['observed_grasp_since'],initial_object_z=hand['source_initial_object_z'])
    plans={r['value']['observation_index']:r['value']for r in rows if r['kind']=='planned_H40'};replayed=0
    for i in range(takeover,641):
        teacher.observe(obs[i]['actual_state'])
        if i<640 and i%8==0:
            p=teacher.plan();expected=np.stack([pose10(T,g)for T,g in zip(p['poses'],p['opening'])])
            np.testing.assert_allclose(expected,plans[i]['rows10'],rtol=0,atol=1e-10);replayed+=1
    if not teacher.candidate_complete or replayed!=(640-takeover)//8:
        raise PermissionError('ACTUAL_EXPERT_SUFFIX_REPLAY_AND_ORIGINAL_FINAL_HOLD_REQUIRED')
    from PIL import Image
    for i in range(takeover,641):
        for name in('body_rgb','wrist_rgb'):
            pin=obs[i]['policy_obs'][name];path=folder/pin['file']
            if path.is_symlink()or not path.resolve().is_relative_to(folder.resolve())or sha(path)!=pin['sha256']or path.stat().st_size!=pin['bytes']:
                raise PermissionError('REAL_SUFFIX_RGB_CHANGED')
            with Image.open(path)as image:
                if image.mode!='RGB'or image.size!=(256,256):raise ValueError('ORIGINAL_DUAL256_SOURCE_RGB')
                image.load()
    suffix=suffix_packet(full,takeover);output.mkdir(parents=True,exist_ok=False)
    name=condition['id']+'_EXPERT_SUFFIX';numeric=output/(name+'.npz')
    with numeric.open('xb')as f:np.savez_compressed(f,**suffix)
    row=dict(id=name,task='BOOK',split='TRAIN',family='LIBERO_ARX5_ACTUAL_LATE_HELD_RECOVERY',
        origin='SOURCE_EXPERT_RECOVERY_SUFFIX_NOT_FULL_TEACHER',windows=len(anchors),
        file=numeric.name,sha256=sha(numeric),raw_root=str(folder.resolve()),raw_sha256=raw_sha256,
        audit_sha256=audit_sha256,terminal_sha256=terminal_sha256,takeover_marker_sha256=sha(marker),
        source_policy_checkpoint_sha256=m['delta_checkpoint_manifest_sha256'],source_revision=m['code_source_sha'],
        original_TRAIN_parent=f"BOOK_SOURCE_SEED{condition['TRAIN_parent']:03}",event=condition['event'],
        source_observation_start=takeover,source_observation_end=640,complete_commands=640-takeover,
        actual_observations=641-takeover,student_prefix_expert_labels=0,source_full_task_success=True,
        full_pure_Teacher_episode=False,independent_new_layout=False,training_admitted=False,
        actual_expert_plans_replayed=replayed,full_handover_snapshot_sha256=hand['full_state']['sha256'],
        new_simulations_models_statistics=0)
    with(output/'SOURCE_PROVENANCE.json').open('x')as f:json.dump(row,f,indent=2);f.write('\n')
    return row
