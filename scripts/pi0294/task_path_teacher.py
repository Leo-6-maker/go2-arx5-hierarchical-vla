"""Approved-S1 task geometry over persistent, planned world EEF10 paths.

No WBC modifications, joint commands, BASE labels or physical admission.
The native worker owns actual feedback, safety, installation and accounting.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from go2arx5_vla.task_data.codec import decode
from .component_teacher import geometry
from .persistent_approach_draft import pose
from .task_gripper_contract import teacher_physical_fraction
from .task_path_model import PlanningObjectMotion, certify_chunk
from .task_path_planning import grasp_precontact_candidates, push_precontact_candidates
from .task_path_reference import PersistentPathReference

VERSION = 'S1_NATIVE_STAGE_PATH_EEF10_V1'


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def executed_reference_history(intervals):
    """Decode submitted EEF10 with the same row convention as pose10/wire.

    These are the last two executed reference values, not achieved TCP labels.
    Column-convention decoding transposes the rotations and invents a jump at
    the next persistent path boundary.
    """
    history = []
    for interval in intervals[-2:]:
        reference, _ = decode(interval['action_EEF10'], np.eye(4))
        history.append(dict(pose=reference, end_native_step=interval['end_native_step']))
    return history


class NativePathClient:
    def __init__(self, *, folder, python, pythonpath, code, model_path, model_sha256, urdf, srdf):
        self.folder = Path(folder); self.folder.mkdir()
        self.python, self.pythonpath, self.code = python, pythonpath, Path(code)
        self.model_path, self.model_sha = str(model_path), model_sha256
        self.urdf, self.srdf, self.calls = str(urdf), str(srdf), 0

    def plan(self, sample, goal, *, stage, motion, direction, speed):
        if self.calls >= 6: raise PermissionError('S1_SIX_NEW_STAGE_PLANS_MAX_NO_REDRAW')
        index = self.calls; self.calls += 1
        request, output = self.folder/f'{index:02d}-{stage}.input.private.json', self.folder/f'{index:02d}-{stage}.result.private.json'
        value = dict(schema='S1_ACTUAL_STAGE_MPLIB_REQUEST_V1', stage=stage,
            model_path=self.model_path, model_sha256=self.model_sha, actual_sample=sample,
            goal_world=goal, urdf=self.urdf, srdf=self.srdf, object_motion=motion,
            push_direction=direction, linear_speed_m_s=speed)
        raw = json.dumps(value, default=serial, sort_keys=True, allow_nan=False).encode()
        with request.open('xb') as f: f.write(raw)
        env = dict(os.environ, PYTHONPATH=str(self.code)+':'+str(self.code/'src'),
            S1_RUNTIME_EXTRA_SITE_PACKAGES=self.pythonpath,
            LD_PRELOAD='', LD_AUDIT='', LD_LIBRARY_PATH='', CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (self.folder/f'{index:02d}-{stage}.log').open('x') as log:
            subprocess.run([self.python, '-B', '-m', 'scripts.pi0294.task_path_backend',
                '--request', str(request), '--output', str(output)], cwd=self.code,
                env=env, check=True, timeout=50, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(output.read_bytes())
        if result['request_sha256'] != hashlib.sha256(raw).hexdigest(): raise ValueError('STAGE_BACKEND_INPUT_BINDING')
        if result['status'] != 'TIMED_STATIC_REFERENCE_PREPARED_NOT_EXECUTED':
            raise ValueError('STAGE_PLANNING_REJECT_PRESERVED:'+result['status'])
        artifact = output.with_suffix('.npz')
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != result['artifact_sha256']:
            raise ValueError('STAGE_BACKEND_ARTIFACT_CHANGED')
        with np.load(artifact, allow_pickle=False) as data: poses = data['poses_world'].copy()
        return poses, result


class Teacher:
    def __init__(self, card, engine, client, *, sequence_offset=0):
        self.card, self.engine, self.client = card, engine, client
        self.sequence_offset = sequence_offset
        self.stage, self.planned_stage, self.reference = 'approach', None, None
        self.events, self.planning = [], []
        self.stop_reason = self.initial = self.precontact = self.goal = self.grasp_since = None
        self.index, self.initial_step = -1, None
        self.direction = None
        self.milestones = dict(position_progress=False, safe_precontact=False, actual_contact=False, manipulation=False)
        self.geometry = None

    def observe(self, sample):
        if self.stop_reason: raise PermissionError(self.stop_reason)
        self.index += 1
        self.sample = sample
        if not sample['safety_valid']:
            self.stop_reason = 'ORIGINAL_NATIVE_SAFETY_STOP'; raise PermissionError(self.stop_reason)
        if self.initial is None:
            self.initial, self.initial_step = pose(sample['object_pose']), sample['native_step']
            if self.card['task'] == 'T1':
                choice = grasp_precontact_candidates(self.initial)[1]  # Fixed NEG30, never result-dependent.
            else:
                pads = []
                for side, g in enumerate(self.engine.pad):
                    tcp = sample['tcp']; center = tcp[:3, :3].T @ (self.engine.data.geom_xpos[g]-tcp[:3, 3])
                    center[1] -= (1 if side == 0 else -1)*sample['finger_q'][side]
                    pads.append(dict(center_tcp=center, axes_tcp=tcp[:3, :3].T @ self.engine.data.geom_xmat[g].reshape(3, 3),
                                     half_size=self.engine.model.geom_size[g].copy()))
                choice = push_precontact_candidates(self.initial, self.card['goal_xy'], pads)[2]
                self.direction = choice['direction']; self.front = choice['front_support_m']
            self.precontact = choice['goal_world']; self.initial_distance = float(np.linalg.norm(sample['tcp'][:3, 3]-self.precontact[:3, 3]))
        if sample['native_step'] != self.initial_step+self.index*25: raise ValueError('ACTUAL_20HZ_TASK_CLOCK')
        if self.reference is not None:
            self.reference.observe(native_step=sample['native_step'], raw_time_s=sample['raw_time_s'],
                actual_tcp=sample['tcp'], native_safe=sample['safety_valid'])
        self.geometry = geometry(self.engine, sample, self.card['task'], self.direction)
        self.milestones['actual_contact'] |= sample['robot_object_contact']
        self.milestones['safe_precontact'] |= self.geometry['precontact_geometry_ready']
        self.milestones['position_progress'] |= self.initial_distance-np.linalg.norm(sample['tcp'][:3, 3]-self.precontact[:3, 3]) >= .02
        if sample['native_grasp']:
            if self.grasp_since is None: self.grasp_since = sample['time_s']
        else: self.grasp_since = None
        displacement = np.linalg.norm(sample['object_pose'][:2, 3]-self.initial[:2, 3])
        lift = sample['object_pose'][2, 3]-self.initial[2, 3]
        if self.card['task'] == 'T1':
            self.milestones['manipulation'] |= self.grasp_since is not None and sample['time_s']-self.grasp_since >= .3
            if self.stage in ('lift', 'carry') and not sample['native_grasp']:
                self.stop_reason = 'ACTUAL_GRASP_LOST_TASK_FAIL'; return
        else:
            self.milestones['manipulation'] |= sample['robot_object_contact'] and not sample['native_grasp'] and displacement >= .01 and lift <= .005
            if lift > .005+1e-8:
                self.stop_reason = 'ACTUAL_PUSH_LIFTED_OBJECT_TASK_FAIL'; return
        if self.index % 8: return
        old = self.stage; s = sample; task = self.card['task']
        goal_error = np.linalg.norm(s['object_pose'][:2, 3]-self.card['goal_xy'])
        if old == 'approach' and self.geometry['precontact_geometry_ready']: self.stage = 'align' if task == 'T1' else 'close'
        elif old == 'align' and self.geometry['actual_contact_alignment']: self.stage = 'close' if task == 'T1' else 'contact'
        elif old == 'close' and task == 'T1' and s['native_grasp']: self.stage = 'verify_grasp'
        elif old == 'verify_grasp' and self.milestones['manipulation']: self.stage = 'lift'
        elif old == 'lift' and lift >= .06 and s['native_grasp']: self.stage = 'carry'
        elif old == 'carry' and goal_error <= .01 and s['native_grasp']: self.stage = 'lower'
        elif old == 'lower' and s['object_table_contact'] and goal_error <= .02: self.stage = 'release'
        elif old == 'release' and not s['native_grasp'] and s['gripper_fraction'] >= .85: self.stage = 'withdraw'
        elif old == 'close' and task == 'T2' and s['gripper_fraction'] <= .05: self.stage = 'align'
        elif old == 'contact' and s['robot_object_contact'] and not s['native_grasp']: self.stage = 'push'
        elif old == 'push' and goal_error <= .02 and displacement >= .10: self.stage = 'detach'
        elif old == 'detach' and not s['robot_object_contact'] and np.linalg.norm(s['tcp'][:3, 3]-self.goal[:3, 3]) <= .01: self.stage = 'withdraw'
        elif old == 'withdraw' and not s['robot_object_contact'] and s['tcp'][2, 3]-self.withdraw_start_z >= .08: self.stage = 'hold'
        if old != self.stage:
            self.events.append(dict(index=self.index, time_s=s['time_s'], previous=old, stage=self.stage,
                basis='ACTUAL_TASK_GEOMETRY_CONTACT_AND_OBJECT_MOTION', geometry=self.geometry))

    def propose(self):
        s = self.sample
        if self.stop_reason: raise PermissionError(self.stop_reason)
        if self.index % 8: raise ValueError('PREFIX8_QUERY_BOUNDARY')
        stage, task = self.stage, self.card['task']
        motion = ('ACTUAL_GRASP_ATTACHMENT' if task == 'T1' and stage in ('lift', 'carry', 'lower') else
                  'ACTUAL_CONTACT_PUSH_TRANSLATION' if task == 'T2' and stage == 'push' else 'STATIC')
        if self.planned_stage != stage:
            goal = self.precontact.copy()
            if stage in ('align', 'contact'):
                if task == 'T1': goal[2, 3] = self.initial[2, 3]+.01
                else:
                    goal[2, 3] -= .08; goal[:2, 3] += self.direction*(.013 if stage == 'align' else .026)
            elif stage in ('lift', 'carry', 'lower'):
                goal[:3, 3] = self.initial[:3, 3]+[0, 0, .09]
                if stage != 'lift': goal[:2, 3] = self.card['goal_xy']
                if stage == 'lower': goal[2, 3] = self.initial[2, 3]+.01
            elif stage == 'push':
                goal[2, 3] -= .08
                support = float(np.abs(self.direction @ self.initial[:2, :3]) @ np.full(3, .02))
                goal[:2, 3] = np.asarray(self.card['goal_xy'])-self.direction*(support+self.front-.001)
            elif stage == 'detach':
                goal = s['tcp'].copy(); goal[:2, 3] -= self.direction*.04
            elif stage == 'withdraw':
                goal = s['tcp'].copy(); self.withdraw_start_z = float(goal[2, 3]); goal[2, 3] += .13
            elif stage in ('close', 'verify_grasp', 'release', 'hold'): goal = s['tcp'].copy()
            hold = stage in ('close', 'verify_grasp', 'release', 'hold')
            if hold: poses = np.repeat(goal[None], 2, axis=0)
            else:
                poses, proof = self.client.plan(s, goal, stage=stage, motion=motion,
                    direction=self.direction, speed=.04 if task == 'T2' and stage in ('contact', 'push', 'detach') else .06)
                self.planning.append(proof)
            logical = stage in ('approach', 'align', 'release', 'withdraw', 'hold') if task == 'T1' else stage in ('approach', 'withdraw', 'hold')
            opening = teacher_physical_fraction(np.full(len(poses), float(logical)))
            if self.reference is None:
                self.reference = PersistentPathReference(episode_id=self.card['id'], poses_world=poses,
                    opening=opening, initial_native_step=self.initial_step, sequence_offset=self.sequence_offset)
                self.reference.observe(native_step=s['native_step'], raw_time_s=s['raw_time_s'],
                    actual_tcp=s['tcp'], native_safe=s['safety_valid'])
            else: self.reference.replace_stage_path(poses_world=poses, opening=opening)
            self.planned_stage, self.goal = stage, goal
        result = self.reference.chunk()
        history = executed_reference_history(self.reference.intervals)
        hypothesis = PlanningObjectMotion(s, mode=motion, push_direction=self.direction)
        check = certify_chunk(self.engine.model, s, np.stack([self.reference.poses[min(
            self.index-self.reference.path_start_index+i, len(self.reference.poses)-1)] for i in range(40)]),
            stage=stage, history=history, planning_object_motion=hypothesis)
        result.update(stage=stage, reference_certificate=check, object_hypothesis=hypothesis.evidence())
        return result
