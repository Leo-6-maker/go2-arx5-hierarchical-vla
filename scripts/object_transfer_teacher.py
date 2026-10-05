"""Draft teacher setpoints from measured tool/object state; no simulator or policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix, rotation_matrix_to_rot6d_rows

STAGES = ('PREGRASP', 'APPROACH', 'CLOSE', 'LIFT', 'TRANSFER', 'LOWER', 'RELEASE', 'RETREAT')
TIMEOUTS = (4., 4., 2., 4., 6., 4., 2., 4.)


def finite(value, shape):
    x = np.asarray(value, float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f'measured finite {shape} required')
    return x


class Teacher:
    """Privileged supervision only. Never call this from learned-policy execution."""
    def __init__(self, source_object, bin_center, bin_floor, object_half_z, initial_tcp, initial_rotation, initial_width):
        source = finite(source_object, (3,))
        center = finite(bin_center, (2,))
        scalar = finite([bin_floor, object_half_z, initial_width], (3,))
        if object_half_z <= 0 or not 0 <= initial_width <= .08:
            raise ValueError('invalid draft tool or object dimensions')
        # Draft tool fingers extend +z from the palm TCP; grasp uses fixed downward R.
        self.rotation = np.diag([1., -1., -1.])
        grasp = source + [0, 0, .02]
        lower = np.r_[center, scalar[0] + scalar[1] + .02]
        self.goals = (grasp + [0, 0, .08], grasp, grasp,
                      grasp + [0, 0, .10], lower + [0, 0, .10], lower, lower, lower + [0, 0, .12])
        self.support = source[2] - object_half_z
        self.half_z = object_half_z
        self.stage = 0
        self.entered = 0.
        self.ready_since = None
        self.last_time = None
        self.command = finite(initial_tcp, (3,)).copy()
        self.command_rotation = finite(initial_rotation, (3, 3)).copy()
        if not np.allclose(self.command_rotation @ self.command_rotation.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(self.command_rotation)-1) > 1e-6:
            raise ValueError('initial measured TCP must be SO3')
        self.width = float(initial_width)
        self.grasp_offset = None
        self.failure = None

    def update(self, observation):
        t = float(finite(observation['time'], ()))
        dt = 0. if self.last_time is None else t - self.last_time
        if (self.last_time is None and abs(t) > 1e-9) or dt < 0 or (self.last_time is not None and (dt <= 0 or dt > 1/120 + 1e-8)):
            raise ValueError('complete monotonic 120Hz teacher clock required')
        p = finite(observation['tcp_position'], (3,))
        r = finite(observation['tcp_rotation'], (3, 3))
        obj = finite(observation['target_position'], (3,))
        bottom, width, speed, penetration = finite([observation[k] for k in ('target_bottom_z', 'measured_width_m', 'target_speed_m_s', 'penetration_m')], (4,))
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(r)-1) > 1e-6:
            raise ValueError('measured TCP must be SO3')
        if not 0 <= width <= .08 or min(speed, penetration) < 0:
            raise ValueError('invalid measured opening, speed or penetration')
        for key in ('bilateral_contact', 'target_tool_contact', 'supported', 'dangerous_contact', 'gate_reject'):
            if not isinstance(observation[key], (bool, np.bool_)):
                raise ValueError('measured contact/gate flags must be booleans')
        both, contact = observation['bilateral_contact'], observation['target_tool_contact']
        if both and not contact:
            raise ValueError('contradictory measured contact')
        if self.last_time is None and (np.linalg.norm(p-self.command) > 1e-6 or geodesic_angle(r, self.command_rotation) > 1e-6 or abs(width-self.width) > 1e-6):
            raise ValueError('first measurement differs from declared teacher initial state')
        self.last_time = t
        if not self.failure:
            if observation['gate_reject'] or observation['dangerous_contact'] or penetration > .002:
                self.failure = 'GATE_OR_DANGEROUS_CONTACT'
            elif self.stage < 8 and (t >= 30 or t - self.entered > TIMEOUTS[self.stage]):
                self.failure = 'TASK_OR_STAGE_TIMEOUT'
            elif self.stage in (3, 4, 5) and (not both or np.linalg.norm(obj-p-self.grasp_offset) > .005 + 1e-9):
                self.failure = 'LOST_GRASP_OR_SLIP'
        if self.stage < 8 and not self.failure:
            pose_ready = np.linalg.norm(p-self.goals[self.stage]) <= .005 and geodesic_angle(r, self.rotation) <= .05
            ready = pose_ready
            if self.stage == 1:
                ready = pose_ready and np.linalg.norm(r.T @ (obj-p) - [0, 0, .02]) <= .005
            elif self.stage == 2: ready = pose_ready and both and 0 < width < .070
            elif self.stage == 3: ready = pose_ready and bottom-self.support >= .08 - 1e-9
            elif self.stage == 4: ready = pose_ready and both
            elif self.stage == 5: ready = pose_ready and observation['supported'] and speed < .02
            elif self.stage == 6: ready = pose_ready and width >= .070 and not contact
            elif self.stage == 7: ready = pose_ready and width >= .070 and not contact and np.linalg.norm(p-obj) >= .10 and speed < .02
            self.ready_since = (t if self.ready_since is None else self.ready_since) if ready else None
            duration = .3 if self.stage == 3 else 2. if self.stage == 7 else .2
            if self.ready_since is not None and t-self.ready_since >= duration - 1e-9:
                if self.stage == 2: self.grasp_offset = (obj-p).copy()
                self.stage += 1
                self.entered = t
                self.ready_since = None
        if not self.failure:
            target = self.goals[min(self.stage, 7)]
            delta = target-self.command
            self.command += delta * min(1., .10*dt/max(np.linalg.norm(delta), 1e-12))
            angle = float(geodesic_angle(self.command_rotation, self.rotation))
            self.command_rotation = quat_to_matrix(quat_slerp(matrix_to_quat(self.command_rotation), matrix_to_quat(self.rotation), min(1., .5*dt/max(angle, 1e-12))))
            opening = .0 if self.stage in (2, 3, 4, 5) else .08
            self.width += float(np.clip(opening-self.width, -.04*dt, .04*dt))
        return dict(commanded_label=np.r_[self.command, rotation_matrix_to_rot6d_rows(self.command_rotation), self.width/.08],
                    evidence=dict(teacher_stage=STAGES[self.stage] if self.stage < 8 else 'DONE', failure=self.failure,
                                  install_allowed=self.failure is None, teacher_finished=self.stage == 8,
                                  task_success='SEPARATE_OBJECT_RESULT_SCORER_REQUIRED'))


def fixture(teacher, time):
    s = min(teacher.stage, 7)
    p = teacher.goals[s].copy()
    source = np.array([-.2, 0, .665])
    obj = source if s < 3 else (p if s < 6 else teacher.goals[5]) + [0, 0, -.02]
    return dict(time=time, tcp_position=p, tcp_rotation=teacher.rotation.copy(), target_position=obj,
                target_bottom_z=obj[2]-.015, measured_width_m=.03 if s in (2, 3, 4, 5) else .08,
                target_speed_m_s=0., penetration_m=0., bilateral_contact=s in (2, 3, 4, 5),
                target_tool_contact=s in (2, 3, 4, 5), supported=s not in (3, 4), dangerous_contact=False, gate_reject=False)


def demo():
    def new(): return Teacher([-.2, 0, .665], [.2, .2], .65, .015, [-.2, 0, .765], np.diag([1., -1., -1.]), .08)
    a = new()
    stages = []
    previous = None
    for i in range(1200):
        result = a.update(fixture(a, i/120))
        label = result['commanded_label']
        if previous is not None:
            assert np.linalg.norm(label[:3]-previous[:3]) <= .10/120 + 1e-12
            assert abs(label[9]-previous[9])*.08 <= .04/120 + 1e-12
        previous = label.copy()
        if not stages or stages[-1] != result['evidence']['teacher_stage']: stages.append(result['evidence']['teacher_stage'])
        if a.stage == 8: break
    assert a.stage == 8 and a.failure is None and stages == list(STAGES)+['DONE']
    b = new()
    b.command_rotation = np.eye(3)
    prev = b.command_rotation.copy()
    for i in range(30):
        measured = fixture(b, i/120)
        measured['tcp_rotation'] = np.eye(3)
        b.update(measured)
        assert geodesic_angle(prev, b.command_rotation) <= .5/120 + 1e-10
        prev = b.command_rotation.copy()
    b = new()
    for i in range(530):
        x = fixture(b, i/120)
        if i: x['tcp_position'] += [1, 0, 0]
        blocked = b.update(x)
    assert b.stage == 0 and blocked['evidence']['failure'] == 'TASK_OR_STAGE_TIMEOUT'
    negatives = {'NO_ACTUAL_APPROACH_STAYS_PREGRASP': True}
    for name in ('NO_BILATERAL_GRASP', 'LOST_GRASP', 'SLIP', 'GATE_REJECT', 'DANGEROUS_CONTACT', 'PENETRATION'):
        b = new()
        for i in range(1000):
            x = fixture(b, i/120)
            if name == 'NO_BILATERAL_GRASP': x['bilateral_contact'] = False
            elif name == 'LOST_GRASP' and b.stage >= 3: x['bilateral_contact'] = x['target_tool_contact'] = False
            elif name == 'SLIP' and b.stage >= 3: x['target_position'] += [.02, 0, 0]
            elif name == 'GATE_REJECT': x['gate_reject'] = True
            elif name == 'DANGEROUS_CONTACT': x['dangerous_contact'] = True
            elif name == 'PENETRATION': x['penetration_m'] = .003
            blocked = b.update(x)
            if b.failure: break
        assert b.failure and not blocked['evidence']['install_allowed'], name
        # Failure is absorbing: healthy observations never re-enter or erase it.
        fail = b.failure
        assert b.update(fixture(b, (i+1)/120))['evidence']['failure'] == fail
        negatives[name] = fail
    b = new()
    corrupt = fixture(b, 0.)
    corrupt['tcp_position'][:] = np.nan
    try: b.update(corrupt)
    except ValueError: pass
    else: raise AssertionError('corrupt state accepted')
    return dict(status='PASS_CPU_DRAFT_TEACHER_TRANSITIONS_BOUNDS_AND_FAILURES', stage_sequence=stages,
                negative_cases=negatives, corrupt_state_rejected=True, synthetic_logic_fixture_only=True,
                teacher_finished_is_not_task_success=True, actual_teacher_episodes=0, teacher_calls=0,
                physics_steps=0, renders=0, model_queries=0, optimizer_updates=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), launch_allowed=False,
                selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
