"""Feedback rolling H40 expert; commit only the actually consumed P10 boundary.

The two cascaded velocity filters have continuous velocity and acceleration.
They retain state across replans, including phase changes. Their 7.5 mm/s
translation bound also retains the old 15 mm maximum H40 displacement. This
is an expert timing intervention, not a learned policy or WBC modification.
"""
import numpy as np

from .task_learning_grasp_v1 import FiftyMsProgram, Teacher
from ..trajectory.so3 import matrix_to_quat, quat_to_matrix, rotation_matrix_to_rot6d_rows

VERSION = 'MOBILE_CONTINUOUS_TEACHER_V2'
DT = .05
TRANSLATION_SPEED = .0075
ANGULAR_SPEED = .02
TRANSLATION_TAU = .15
ANGULAR_TAU = .3


def bounded_velocity(error, maximum):
    value = np.asarray(error, float) / .5
    length = np.linalg.norm(value)
    return value * min(1., maximum / length) if length else value


def advance_filters(position, velocity, inner_velocity, command, tau):
    """Exact integration for a constant command over one 50ms interval."""
    decay = np.exp(-DT / tau)
    d = inner_velocity - command
    displacement = (command * DT + (velocity-command) * tau * (1-decay)
                    + d * tau * (1-(1+DT/tau)*decay))
    new_velocity = command + (velocity-command+d*DT/tau) * decay
    new_inner = command + d * decay
    return position+displacement, new_velocity, new_inner


def rotation_vector(rotation):
    q = matrix_to_quat(rotation)
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return q[:3] * (2*np.arctan2(length, q[3])/length) if length > 1e-14 else np.zeros(3)


def rotation_increment(vector):
    angle = np.linalg.norm(vector)
    q = np.r_[vector*(np.sin(angle/2)/angle), np.cos(angle/2)] if angle > 1e-14 else np.array([0.,0.,0.,1.])
    return quat_to_matrix(q)


class ContinuousTeacher(Teacher):
    version = VERSION

    def __init__(self, initial):
        super().__init__(initial)
        self.initial_cube_z = float(self.cube[2,3])
        self.velocity = np.zeros(3)
        self.inner_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.inner_angular_velocity = np.zeros(3)
        self.close_start = None
        self.last_committed_step = None
        self.pending = None

    def goal(self, sample):
        actual = np.asarray(sample['tcp'])
        # Geometry is unchanged; use current object feedback before contact.
        if self.stage in ('PREGRASP','DESCEND','CLOSE'):
            self.cube = np.asarray(sample['object_pose']).copy()
        goal = self.cube[:3,3] + [0,-.0017,.080]
        if self.stage == 'PREGRASP' and np.linalg.norm(actual[:3,3]-goal) <= .004:
            self.stage = 'DESCEND'
        if self.stage == 'DESCEND':
            goal = self.grasp_goal()
            if np.linalg.norm(actual[:3,3]-goal) <= .004:
                self.stage = 'CLOSE'
        if self.stage == 'CLOSE':
            goal = self.grasp_goal()
            if self.close_start is None:
                self.close_start = sample['native_step']
            if sample.get('native_grasp') is True:
                self.lift_origin = actual[:3,3].copy()
                self.stage = 'LIFT'
        if self.stage in ('LIFT','HOLD'):
            goal = self.lift_origin + [0,0,.026]
            if sample['object_pose'][2,3]-self.initial_cube_z >= .020:
                self.stage = 'HOLD'
        return goal

    def grip_trajectory(self, start):
        if self.close_start is None:
            return np.ones(40)
        u = np.clip(((start-self.close_start)*.002+np.arange(40)*DT)/2.,0.,1.)
        return 1-u**3*(10+u*(-15+6*u))

    def decide(self, sample, reference, base):
        actual, reference = np.asarray(sample['tcp']), np.asarray(reference)
        start = sample['native_step']
        if self.last_committed_step is not None and start != self.last_committed_step:
            raise ValueError('CONTINUOUS_TEACHER_REQUIRES_EXACT_CONSUMED_P10_CLOCK')
        goal = self.goal(sample)
        p, r = reference[:3,3].copy(), reference[:3,:3].copy()
        v, z = self.velocity.copy(), self.inner_velocity.copy()
        av, az = self.angular_velocity.copy(), self.inner_angular_velocity.copy()
        origin = p.copy()
        rotation_map = reference[:3,:3] @ actual[:3,:3].T
        path, states = [], []
        for k in range(40):
            path.append((p.copy(), r.copy()))
            states.append((v.copy(),z.copy(),av.copy(),az.copy()))
            # Every real replan uses actual TCP/object feedback. Preview uses
            # the current measured tracking offset, never future achieved state.
            predicted_actual = actual[:3,3] + p-origin
            command = bounded_velocity(goal-predicted_actual, TRANSLATION_SPEED)
            p, v, z = advance_filters(p,v,z,command,TRANSLATION_TAU)
            predicted_rotation = actual[:3,:3] @ reference[:3,:3].T @ r
            angular_error = rotation_map @ rotation_vector(self.rotation @ predicted_rotation.T)
            angular_command = bounded_velocity(angular_error, ANGULAR_SPEED)
            increment,av,az = advance_filters(np.zeros(3),av,az,angular_command,ANGULAR_TAU)
            r = rotation_increment(increment) @ r
        raw = np.empty((40,10))
        raw[:,:3] = actual[:3,3] + (np.asarray([p for p,r in path])-origin)
        raw[:,3:9] = rotation_matrix_to_rot6d_rows(np.asarray([
            actual[:3,:3] @ reference[:3,:3].T @ r for p,r in path]))
        raw[:,9] = self.grip_trajectory(start)
        program = FiftyMsProgram(raw,actual,reference,base)
        if np.linalg.norm(raw[-1,:3]-raw[0,:3]) > .015+1e-12:
            raise ValueError('CONTINUOUS_TEACHER_RETAINS_15MM_H40_SPAN')
        self.pending = (program,start,states)
        return program,dict(stage=self.stage,privileged_goal=goal.tolist(),
            submitted_policy_H40=raw.tolist(),teacher_version=VERSION,
            continuous_reference_start=reference.tolist(),
            initial_velocity_m_s=self.velocity.tolist(),
            initial_acceleration_m_s2=((self.inner_velocity-self.velocity)/TRANSLATION_TAU).tolist(),
            stage_progress_not_reset=True,maximum_H40_translation_m=.015,
            no_future_achieved_state_labels=True,student_metadata=False)

    def commit(self, program, native_step):
        if self.pending is None or self.pending[0] is not program or native_step != self.pending[1]+250:
            raise ValueError('COMMIT_ONLY_REAL_COMPLETE_P10')
        self.velocity,self.inner_velocity,self.angular_velocity,self.inner_angular_velocity = (
            a.copy() for a in self.pending[2][10])
        self.last_committed_step = native_step
        self.pending = None
        # The next query occurs at t=0.5s, the wire's point 10. Row 9 was
        # issued at t=0.45s and must not reset this continuous reference.
        from scripts.pi0292.eval_contract import matrix10
        return matrix10(program.preview(10)[0])
