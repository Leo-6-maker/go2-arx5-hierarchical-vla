"""Replay original Teacher decisions on saved TRAIN observations, not physics."""
from copy import deepcopy
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.data_cov1.export import event_index
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.scene_cards import digest
from .scope import sealed_file, validate


def replay(card, initial, samples, physics, references, events, push_geometry=None):
    """Geometry is the recorded native measurement, not a new geometry model."""
    from scripts.pi0291_arx5_data.teacher import Teacher as PickTeacher
    from scripts.pi0294.teacher import PushTeacher
    if card.get('split') != 'TRAIN' or card.get('task') not in ('T1','T2'):
        raise PermissionError('VAP1_T0_ONLY_EXPLICIT_SAVED_TRAIN')
    initial = np.asarray(initial, dtype=float)
    if card['task'] == 'T1':
        teacher = PickTeacher(card, initial)
    else:
        if push_geometry is None:
            raise PermissionError('VAP1_T0_SAVED_MEASURED_PAD_GEOMETRY_REQUIRED')
        with patch('scripts.pi0294.teacher.push_geometry', return_value=deepcopy(push_geometry)):
            teacher = PushTeacher(card, initial, None)
    if len(samples) != len(physics['time_s']):
        raise ValueError('VAP1_T0_EXACT_SAVED_OBSERVATION_CLOCK')
    reference_count = 0
    for i, sample in enumerate(samples):
        obs = sample['policy_obs']; privileged = sample['privileged']
        if (sample['causal_history']['episode_id'] != card['id']
                or abs(float(physics['time_s'][i])-obs['timestamp_s']) > 1e-8):
            raise ValueError('VAP1_T0_SAVED_EPISODE_OR_CLOCK_MISMATCH')
        state = {k: physics[k][i] for k in ('time_s','object_pose','native_grasp',
            'left_object_contact','right_object_contact','robot_object_contact')}
        state.update(tcp=np.asarray(sample['execution']['actual_tcp']),gripper_fraction=obs['actual_gripper'])
        teacher.observe(state)
        if (teacher.stage != privileged['teacher_stage'] or bool(teacher.lifted) != privileged['ever_lifted']
                or bool(teacher.carried) != privileged['ever_carried']):
            raise ValueError('VAP1_T0_ORIGINAL_FEEDBACK_STAGE_REPLAY_MISMATCH')
        if i%8 == 0 and i < len(samples)-1:
            plan=teacher.plan(state); saved=references[reference_count]
            if (saved['decision_index'] != i//8 or saved['privileged_teacher_stage'] != plan['stage']
                    or np.asarray(saved['planned_H40']).shape != (40,4,4)
                    or not np.allclose(plan['poses'],saved['planned_H40'],rtol=0,atol=1e-12)
                    or not np.array_equal(plan['opening'],saved['gripper_H40'])):
                raise ValueError('VAP1_T0_ORIGINAL_SAVED_H40_REPLAY_MISMATCH')
            reference_count += 1
    if teacher.events != events or reference_count != len(references):
        raise ValueError('VAP1_T0_ALL_ORIGINAL_EVENTS_AND_REFERENCES_ACCOUNTED')
    return dict(status='PASS_SAVED_TRAIN_ORIGINAL_TEACHER_EVENT_AND_H40_REPLAY',
        observations=len(samples),references=reference_count,events=events,
        actual_event_index=event_index(physics,card),
        origin='TEACHER_ON_MANIFOLD_NOT_ACTUAL_STUDENT',
        new_physics=0,new_model_forwards=0,recovery_qualification=False)


def run(root):
    from scripts.pi0294.cr2.runtime import bind_cpu
    r=Path(root);m,_=validate(r,'EXPORT');work=read(r/'plans/WORK_PLAN.private.json')
    if (work.get('operation') != 'VAP1_SAVED_TRAIN_TEACHER_EVENT_AUDIT'
            or not 0 < len(work.get('cases',[])) <= 400):
        raise PermissionError('VAP1_T0_EXPLICIT_BOUNDED_SAVED_TRAIN_PLAN')
    lock=OwnerLock(r/'T0_SAVED_TRAIN_OWNER.lock');bind_cpu(m)
    ledger=read(r/'RESOURCE_LEDGER.json');assert ledger['status']=='RESERVED' and not ledger['unknown_effects']
    began=time.process_time();failure=None;rows=[]
    def guard():
        if (time.time() >= m['limits']['operation_deadline_epoch']
                or time.process_time()-began >= m['limits']['CPU_s']-30
                or any((p/'STOP.json').exists() for p in (r,r.parent.parent))):
            raise PermissionError('VAP1_T0_BOUNDED_SAVED_DATA_ONLY')
    try:
        ledger['status']='RUNNING';write(r/'RESOURCE_LEDGER.json',ledger)
        for case in work['cases']:
            guard();folder=Path(case['source_episode']);b=read(folder.parent.parent/'OUTPUT_BINDING.private.json')
            if read(b['lease'])['status']!='CLOSED' or sha(folder/'FILE_MANIFEST.json')!=case['raw_manifest_sha256']:
                raise PermissionError('VAP1_T0_EXACT_CLOSED_TRAIN_RAW')
            inventory={x['path']:x['sha256'] for x in read(folder/'FILE_MANIFEST.json')}
            names=['CARD.json','INITIAL_STATE.json','samples.json','physics.npz','references.json','TEACHER_EVENTS.json']
            if (folder/'PUSH_GEOMETRY.json').exists():names.append('PUSH_GEOMETRY.json')
            for name in names:
                if name not in inventory or sha(folder/name)!=inventory[name]:
                    raise ValueError('VAP1_T0_RELEVANT_RAW_MEMBER_CHANGED')
            with np.load(folder/'physics.npz',allow_pickle=False) as archive:
                physics={k:archive[k].copy() for k in archive.files}
            card=read(folder/'CARD.json')
            value=replay(card,read(folder/'INITIAL_STATE.json')['object_pose'],read(folder/'samples.json'),
                physics,read(folder/'references.json'),read(folder/'TEACHER_EVENTS.json'),
                read(folder/'PUSH_GEOMETRY.json') if 'PUSH_GEOMETRY.json' in names else None)
            value.update(id=card['id'],task=card['task'],raw_manifest_sha256=case['raw_manifest_sha256'],
                         relevant_member_sha256={n:inventory[n] for n in names})
            put(r/'evidence'/(card['id']+'-T0.json'),value);rows.append(value)
        result=dict(status='T0_SAVED_TEACHER_DISTRIBUTION_REPLAY_COMPLETE_NOT_RECOVERY',
            parents=len(rows),rows=rows,new_physics=0,new_models=0,new_updates=0,
            source_teacher_modified=False,actual_student_recovery_qualified=False)
        put(r/'evidence/T0_TEACHER_REPLAY_RESULT.private.json',result)
        return dict(status=result['status'],parents=len(rows),new_physics=0)
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        put(r/'evidence/T0_TEACHER_REPLAY_FAILURE.private.json',dict(error=failure,parents_completed=len(rows)))
        raise
    finally:
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_EXPORT',CPU_s=time.process_time()-began,
            CPU_reserved_s=0,completed=len(rows),failure=failure)
        write(r/'RESOURCE_LEDGER.json',ledger);lock.close()
