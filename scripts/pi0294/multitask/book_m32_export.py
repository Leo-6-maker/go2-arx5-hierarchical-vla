"""Versioned maintenance TRAIN15 plus actual expert recovery suffixes.

Uses the existing source packet, NPZ reader and official Processor codec.
The original six admitted arrays and DEV seal remain immutable. Admission
requires native CPU qualification of every window under the new statistics.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.train_adapter import DOMAIN, TAG, OFFICIAL_SHA, sha
from . import book_early_recovery as recovery
from .book_seed_population import plan as population_plan
from .libero_book_teacher import teacher_for_card, seed_card
from .recovery_export import expert_suffix_packet, raw, SEAL
from .source_export import export, packet, statistics, KINDS

PARENT_DATA='d3f4999809a9f57288fe5ba62116b70cc8660cff2fb70821d583b81834039d31'
PARENT_STATS='73cc6e2f903f8e56e2de6384bb16d9f4eb9ff98a3a28acd74f1dd0d1b5b80763'
SCHEMA='BOOK_M32_TRAIN15_SIX_EXPERT_SUFFIXES_V1'
COUNTS=(601,)*15+(569,593,585,593,585,593)


def fit_statistics(packets, counts=COUNTS):
    if len(packets)!=len(counts):raise ValueError('EXACT_DECLARED_TRAIN_POPULATION')
    states,actions,relative=[],[],[]
    for p,count in zip(packets,counts):
        s,c=p['actual_state10'][:count],p['committed_absolute10']
        if (s.shape!=(count,10)or c.shape!=(count+39,10)
                or not np.array_equal(p['valid_window'],np.arange(count+39)<count)):
            raise ValueError('COMPLETE_ACTUAL_EXPERT_H40_WINDOWS_ONLY')
        future=np.arange(count)[:,None]+np.arange(40)
        a=c[future];states.append(s);actions.append(a)
        relative.append(pose10(np.linalg.inv(matrix_from10(s))[:,None]@matrix_from10(c)[future],a[...,9])[...,:9])
    s,a,z=np.concatenate(states),np.concatenate(actions),np.concatenate(relative)
    if not all(np.isfinite(v).all()for v in(s,a,z)):raise ValueError('FINITE_UNCLIPPED_EXPERT_LABELS')
    return {TAG:dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:],(0,1))),
        relative_action=dict(eef_9d=statistics(z,(0,1))))}


def _read_packet(root,row):
    file=root/row['file']
    if (not file.resolve().is_relative_to(root.resolve())or file.is_symlink()
            or sha(file)!=row['sha256']or row['split']!='TRAIN'):
        raise ValueError('PINNED_TRAIN_NUMERIC_ONLY')
    with np.load(file,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}


def _verify_saved_images(row,arrays):
    from PIL import Image
    root=Path(row['raw_root'])
    for paths,pins in zip(arrays['rgb_paths'],arrays['rgb_sha256']):
        for name,pin in zip(paths,pins):
            image=root/str(name)
            if Path(str(name)).name!=name or image.is_symlink()or sha(image)!=str(pin):
                raise ValueError('IMMUTABLE_PARENT_EXPERT_RGB_CHANGED')
            with Image.open(image)as im:
                if im.mode!='RGB'or im.size!=(256,256):raise ValueError('ACTUAL_256_RGB')
                im.load()


def early_suffix(folder,condition,expected):
    """Recheck the saved closed producer, complete execution and exact handover."""
    folder=Path(folder)
    if condition not in recovery.conditions()[:5] or folder.name!=condition['id']:
        raise PermissionError('ONLY_THE_FIVE_ACTUALLY_SUCCESSFUL_TRAIN_RECOVERIES')
    for name,pin in expected.items():
        if Path(name).is_absolute()or'..'in Path(name).parts or sha(folder/name)!=pin:
            raise ValueError('PINNED_RECOVERY_MEMBER_CHANGED:'+name)
    if set(expected)!={'CARD.json','INDEPENDENT_RAW_AUDIT.json','TERMINAL.json','ACTUAL_RECORDS.jsonl',
                      f"predictions/{condition['takeover_index']//8:02}_expert_takeover.json"}:
        raise ValueError('COMPLETE_RECOVERY_CAPTURE_PINS')
    card=json.loads((folder/'CARD.json').read_bytes());audit=json.loads((folder/'INDEPENDENT_RAW_AUDIT.json').read_bytes())
    if (card!=recovery.recovery_card(condition)or audit['integrity_problems']
            or audit['raw_sha256']!=expected['ACTUAL_RECORDS.jsonl']
            or audit['terminal_sha256']!=expected['TERMINAL.json']
            or any(audit[k]is not True for k in('task_success','complete_episode','original_native_safety_pass',
                   'native_LIBERO_goal_final','additional_support_release_withdraw_hold_pass'))):
        raise PermissionError('FULL_SAFE_ACTUAL_TRAIN_EXPERT_RECOVERY_REQUIRED')
    rows=[r for r in map(json.loads,(folder/'ACTUAL_RECORDS.jsonl').open())if r['kind']in KINDS]
    full=packet(rows,card);t=condition['takeover_index']
    hand=json.loads((folder/f'predictions/{t//8:02}_expert_takeover.json').read_bytes())
    if (hand['index']!=t or hand['student_prefix_is_expert']is not False
            or hand['original_TRAIN_parent']!=card['original_TRAIN_parent']):
        raise ValueError('ACTUAL_EXPERT_BOUNDARY_NO_STUDENT_RELABELING')
    np.testing.assert_array_equal(hand['actual_state_eef10'],full['actual_state10'][t])
    obs={r['value']['index']:r['value']for r in rows if r['kind']=='observation'}
    plans={r['value']['observation_index']:r['value']for r in rows if r['kind']=='planned_H40'}
    teacher=teacher_for_card(recovery.original_card(condition));replayed=0
    from PIL import Image
    for i in range(t,641):
        teacher.observe(obs[i]['actual_state'])
        if i<640 and i%8==0:
            p=teacher.plan();expected_action=np.stack([pose10(T,g)for T,g in zip(p['poses'],p['opening'])])
            np.testing.assert_allclose(expected_action,plans[i]['rows10'],rtol=0,atol=1e-10);replayed+=1
        for name in('body_rgb','wrist_rgb'):
            pin=obs[i]['policy_obs'][name];image=folder/pin['file']
            if image.is_symlink()or image.stat().st_size!=pin['bytes']or sha(image)!=pin['sha256']:
                raise ValueError('ACTUAL_EXPERT_RGB_CHANGED')
            with Image.open(image)as im:
                if im.mode!='RGB'or im.size!=(256,256):raise ValueError('ACTUAL_256_RGB')
                im.load()
    if not teacher.candidate_complete or replayed!=80-t//8:raise ValueError('COMPLETE_EXPERT_REPLAY')
    return expert_suffix_packet(full,t),card,audit,replayed


def prepare(parent,additional,early,output,*,producer_pins,population):
    parent,output=Path(parent),Path(output)
    if output.exists():raise FileExistsError('PRESERVE_PRIOR_DATA_VERSIONS')
    if (sha(parent/'DATASET_MANIFEST.json')!=PARENT_DATA or sha(parent/'statistics.json')!=PARENT_STATS
            or sha(parent/'TRAIN_DEV_SEAL.json')!=SEAL or population!=population_plan()):
        raise ValueError('EXACT_PARENT_AND_PREDECLARED_TRAIN_DEV_CONFIRM_POPULATION')
    old=json.loads((parent/'DATASET_MANIFEST.json').read_bytes())
    if not old['training_admitted']or [r['id']for r in old['episodes']]!=[
            *[f'BOOK_SOURCE_SEED{i:03}'for i in range(5)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX']:
        raise ValueError('ORIGINAL_FIVE_PLUS_ONE_ADMITTED_PARENT_ONLY')
    if len(additional)!=10 or len(early)!=5:raise ValueError('EXACT_TEN_NEW_FULL_AND_FIVE_SAFE_RECOVERIES')
    run=Path(early[0]['folder']).parent.parent
    required={'RUN_MANIFEST.json','control/LEASE.json','RESULT.json','STOP.json'}
    if set(producer_pins)!=required or any(sha(run/n)!=h for n,h in producer_pins.items()):
        raise ValueError('PINNED_CLOSED_PARTIAL_RECOVERY_PRODUCER')
    lease=json.loads((run/'control/LEASE.json').read_bytes());result=json.loads((run/'RESULT.json').read_bytes())
    producer=json.loads((run/'RUN_MANIFEST.json').read_bytes())
    if (lease['status']!='CLOSED'or lease['result_sha256']!=producer_pins['RESULT.json']
            or lease['manifest_sha256']!=producer_pins['RUN_MANIFEST.json']
            or producer['kind']!=recovery.KIND or producer['recovery_plan']!=recovery.plan()
            or result['status']!='STOPPED_PARTIAL_EARLY_TRAIN_RECOVERY_PANEL'
            or result['evaluated_checkpoint_manifest_sha256']!=recovery.CHECKPOINT
            or result['formal_DEV_conditions']!=0 or result['successes']!=5
            or len(result['conditions'])!=6 or not result['conditions'][-1]['hard_native_safety']):
        raise PermissionError('RETAIN_ACTUAL_PARTIAL_PANEL_AND_HARD_STOP')
    output.mkdir(parents=True);(output/'episodes').mkdir();rows=[];packets=[]
    def append(row,arrays,original=None):
        numeric=output/'episodes'/(row['id']+'.npz')
        if original is None:np.savez_compressed(numeric,**arrays)
        else:
            shutil.copyfile(original,numeric)
            if sha(numeric)!=row['sha256']:raise ValueError('IMMUTABLE_PARENT_NUMERIC_COPY')
        rows.append(dict(row,file=numeric.relative_to(output).as_posix(),sha256=sha(numeric)));packets.append(arrays)
    for row in old['episodes'][:5]:
        arrays=_read_packet(parent,row);_verify_saved_images(row,arrays);append(row,arrays,parent/row['file'])
    for i,case in enumerate(additional,5):
        candidate=output/f'full_source_{i:03}'
        m=export(case['folder'],candidate,**case['pins']);row,=m['episodes']
        if (row['id']!=f'BOOK_SOURCE_SEED{i:03}'
                or json.loads((Path(case['folder'])/'CARD.json').read_bytes())!=seed_card(i)):
            raise ValueError('DECLARED_FULL_SOURCE_ID_AND_ACTUAL_INITIAL_CONDITION')
        append(row,_read_packet(candidate,row))
    row=old['episodes'][5];arrays=_read_packet(parent,row);_verify_saved_images(row,arrays)
    append(row,arrays,parent/row['file']);replayed=0
    for condition,case,actual in zip(recovery.conditions()[:5],early,result['conditions'][:5]):
        if (Path(case['folder']).parent.parent!=run or actual['id']!=condition['id']or not actual['success']
                or actual['raw_sha256']!=case['pins']['ACTUAL_RECORDS.jsonl']):
            raise ValueError('ACTUAL_TRAIN_RECOVERY_RESULT_AND_CAPTURE_IDENTITY')
        arrays,card,audit,n=early_suffix(case['folder'],condition,case['pins']);t=condition['takeover_index'];replayed+=n
        append(dict(id=condition['id']+'_EXPERT_SUFFIX',task='BOOK',family='LIBERO_ARX5_LEFT_PORT',
            origin='SOURCE_EXPERT_RECOVERY_SUFFIX_NOT_FULL_TEACHER',split='TRAIN',windows=601-t,
            raw_root=case['folder'],raw_sha256=audit['raw_sha256'],audit_sha256=case['pins']['INDEPENDENT_RAW_AUDIT.json'],
            instruction=card['instruction'],original_TRAIN_parent=card['original_TRAIN_parent'],
            source_observation_start=t,source_observation_end=640,complete_commands=640-t,
            actual_observations=641-t,student_prefix_expert_labels=0,source_full_task_success=True,
            full_pure_Teacher_episode=False,independent_new_layout=False),arrays)
    if tuple(r['windows']for r in rows)!=COUNTS or replayed!=393:raise ValueError('EXACT_EXECUTED_EXPERT_POPULATION')
    def put(n,v):
        with(output/n).open('xb')as f:f.write(raw(v))
    put('statistics.json',fit_statistics(packets));shutil.copyfile(parent/'TRAIN_DEV_SEAL.json',output/'TRAIN_DEV_SEAL.json')
    put('TRAIN_POPULATION.json',dict(plan=population,ids=[r['id']for r in rows],windows=list(COUNTS),
        original_DEV_seal_sha256=SEAL,confirmation_used_for_fit_or_selection=False,
        producer_pins=producer_pins,early_capture_pins=early,full_capture_pins=additional))
    put('STATISTICS_PROVENANCE.json',dict(rule='M32_DECLARED_ACTUAL_EXPERT_FULL_H40_EQUAL_WINDOW_TIMESTEP_MINMAX',
        state_anchor_rows=sum(COUNTS),action_window_timestep_rows=40*sum(COUNTS),
        student_prefix_or_failed_or_DEV_or_CONFIRM_rows=0,parent_data_sha256=PARENT_DATA,
        parent_statistics_sha256=PARENT_STATS,old_data_statistics_or_R0_overwritten=False))
    m=dict(schema=SCHEMA,domain=DOMAIN,purpose='MAINTENANCE_MATCHED_DATA_CAPACITY_COMPARISON',
        training_admitted=False,official_source_sha=OFFICIAL_SHA,episodes=rows,
        statistics_sha256=sha(output/'statistics.json'),TRAIN_DEV_seal_sha256=SEAL,
        TRAIN_population_sha256=sha(output/'TRAIN_POPULATION.json'),TRAIN_windows=sum(COUNTS),DEV_windows=0,
        DEV_conditions=10,unexposed_confirmation_conditions=20,qualified_pure_Teacher_episodes=15,
        mixed_student_expert_recovery_episodes=6,expert_suffix_windows=3518,student_prefix_expert_labels=0,
        distinct_initial_conditions=15,scene_geometry_families=1,actual_new_expert_H40s_replayed=replayed,
        new_models_queries_optimizer_physics=0,original_six_packets_preserved=True)
    put('DATASET_MANIFEST.json',m);return m,packets


def admit(root):
    root=Path(root);m=json.loads((root/'DATASET_MANIFEST.json').read_bytes())
    q=json.loads((root/'OFFICIAL_QUALIFICATION.json').read_bytes())
    if (m['schema']!=SCHEMA or m['training_admitted']is not False or m['TRAIN_windows']!=sum(COUNTS)
            or q['candidate_manifest_sha256']!=sha(root/'DATASET_MANIFEST.json')
            or q['statistics_sha256']!=sha(root/'statistics.json')
            or q['TRAIN_population_sha256']!=sha(root/'TRAIN_POPULATION.json')
            or m['statistics_sha256']!=q['statistics_sha256']or m['TRAIN_population_sha256']!=q['TRAIN_population_sha256']
            or q['all_loader_anchors_checked']!=sum(COUNTS)or q['all_expert_RGB_verified']is not True
            or q['CUDA_initialized']is not False or q['new_models_queries_optimizer_physics_PBS']!=0
            or len(q['codecs'])!=len(COUNTS)or len(q['actual_processor_batches'])!=len(COUNTS)
            or [c['H40_windows_actually_checked']for c in q['codecs']]!=list(COUNTS)
            or any(c['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                   or c['native_inverse_clip_components_float32']!=0 for c in q['codecs'])):
        raise PermissionError('ACTUAL_ALL_WINDOW_OFFICIAL_PROCESSOR_CODEC_UNDER_NEW_STATISTICS')
    out=root/'admitted';out.mkdir(exist_ok=False);(out/'episodes').mkdir()
    for row in m['episodes']:
        _read_packet(root,row);shutil.copyfile(root/row['file'],out/row['file'])
        if sha(out/row['file'])!=row['sha256']:raise ValueError('ADMITTED_NUMERIC_COPY_CHANGED')
    for name in('statistics.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','STATISTICS_PROVENANCE.json'):
        shutil.copyfile(root/name,out/name)
    m.update(training_admitted=True,official_qualification_sha256=sha(root/'OFFICIAL_QUALIFICATION.json'),
        original_candidate_manifest_sha256=sha(root/'DATASET_MANIFEST.json'),raw_audit_flags_unchanged=True)
    with(out/'DATASET_MANIFEST.json').open('xb')as f:f.write(raw(m))
    return out,m
