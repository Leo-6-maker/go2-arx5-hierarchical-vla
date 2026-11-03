"""Thin wrapper around the unchanged PI0294 fixed-base source driver."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import random
import numpy as np

from .contracts import FRAME, TCP, validate_atom

MEMORY = ('arm_targets', 'finger_targets', 'steps', 'safety', 'last_ik_error',
          '_ik_target', '_cached_ik_solution', '_cached_ik_error', 'ik_solve_calls', 'ik_cache_hits')


class NativeAdapter:
    def __init__(self, card, asset_root, *, executor_id, seed=1103):
        import mujoco
        from scripts.pi0294.scene import make_engine
        from scripts.pi0294.teacher import make_teacher
        if card.get('task') != 'T1' or card.get('domain') != 'FIXED_BASE_ARX5_TASK_SIM':
            raise ValueError('WM01 V1 binds only the fixed-base source T1')
        if card['split'] not in ('TRAIN', 'CALIB', 'DEV', 'TEST') or not executor_id:
            raise ValueError('Explicit split and executor required')
        self.mj = mujoco
        self.engine = make_engine(card, asset_root)
        if (self.engine.model.opt.timestep != .002 or self.engine.model.nu != 8
                or self.engine.model.neq or self.engine.model.nmocap
                or hasattr(self.engine, 'finger_driver')):
            self.engine.close()
            raise ValueError('Unexpected native executor; do not substitute WBC/driver')
        self.card = deepcopy(card)
        self.executor_id = executor_id
        self.rng = np.random.default_rng(seed)
        self.teacher = make_teacher(card, self.engine.snapshot()['object_pose'], self.engine)
        self.history = []
        self.last_command = None
        self.segment_id = 0
        self.model_sha256 = hashlib.sha256(self.engine.xml.encode()).hexdigest()
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.forecast = None
        self.forecast_index = 0
        self.shadow_ticks = 0
        self.max_shadow_error = 0.
        self.teacher.observe(self.snapshot())

    def snapshot(self):
        from scripts.pi0294.episode import _robot_contact
        state = self.engine.snapshot()
        state['robot_object_contact'] = _robot_contact(self.engine, state)
        return deepcopy(state)

    def _get_data(self, data):
        value = np.empty(self.mj.mj_stateSize(self.engine.model, self.state_spec))
        self.mj.mj_getState(self.engine.model, data, value, self.state_spec)
        return dict(integration=value, warnings=[(w.number, w.lastinfo) for w in data.warning])

    def _set_data(self, data, saved):
        value = np.asarray(saved['integration'], dtype=float)
        if value.shape != (self.mj.mj_stateSize(self.engine.model, self.state_spec),) or not np.isfinite(value).all():
            raise ValueError('Invalid complete MuJoCo integration state')
        self.mj.mj_setState(self.engine.model, data, value, self.state_spec)
        self.mj.mj_forward(self.engine.model, data)
        # Forward recomputes derived contacts/forces, but must not replace the
        # saved warmstart, applied controls, mocap, activation or plugin state.
        self.mj.mj_setState(self.engine.model, data, value, self.state_spec)
        for warning, (number, lastinfo) in zip(data.warning, saved['warnings']):
            warning.number, warning.lastinfo = number, lastinfo

    def capture(self):
        """Serializable dynamic state, controller/IK/RNG and score history."""
        return deepcopy(dict(schema='WM01_NATIVE_RESTORE_V1', model_sha256=self.model_sha256,
            card=self.card, executor_id=self.executor_id, segment_id=self.segment_id,
            native=self._get_data(self.engine.data), ik_native=self._get_data(self.engine.ik_data),
            controller={k: getattr(self.engine, k) for k in MEMORY},
            teacher=vars(self.teacher), rng=self.rng.bit_generator.state,
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            history=self.history, last_command=self.last_command,
            forecast=self.forecast, forecast_index=self.forecast_index))

    def restore(self, saved):
        if (saved.get('schema') != 'WM01_NATIVE_RESTORE_V1' or saved['model_sha256'] != self.model_sha256
                or saved['card'] != self.card or saved['executor_id'] != self.executor_id
                or set(saved['controller']) != set(MEMORY)):
            raise ValueError('Native restore identity/controller mismatch')
        self._set_data(self.engine.data, saved['native'])
        self._set_data(self.engine.ik_data, saved['ik_native'])
        for key, value in saved['controller'].items():
            setattr(self.engine, key, deepcopy(value))
        self.teacher.__dict__.clear()
        self.teacher.__dict__.update(deepcopy(saved['teacher']))
        self.rng.bit_generator.state = deepcopy(saved['rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        self.history = deepcopy(saved['history'])
        self.last_command = deepcopy(saved['last_command'])
        self.segment_id = saved['segment_id']
        self.forecast = deepcopy(saved['forecast'])
        self.forecast_index = saved['forecast_index']

    def preflight(self, plan, guard):
        """Reuse the existing S1 copied-state rejection without changing actions."""
        from scripts.pi0294.s1_preflight import predict, bind_install
        nonce = f"WM01:{self.card['id']}:{self.engine.steps}"
        def charge(count, *, invocation):
            self.shadow_ticks += count
        result, expected = predict(self.engine, plan['poses'][:8], plan['opening'][:8],
                                   nonce=nonce, guard=guard, charge=charge)
        self.forecast = None
        self.forecast_index = 0
        if result['decision'] == 'PASS':
            bind_install(self.engine, plan['poses'][:8], plan['opening'][:8], result, nonce)
            self.forecast = expected
        return result

    def teacher_plan(self):
        state = self.snapshot()
        plan = self.teacher.plan(state)
        from ..task_data.codec import pose10
        plan['rows10'] = np.stack([pose10(p, g) for p, g in zip(plan['poses'], plan['opening'])])
        plan['frame'], plan['tcp'] = FRAME, TCP
        return plan

    def gate(self, plan):
        from scripts.pi0292.eval_contract import gate_action
        from ..task_data.codec import pose10
        state = self.snapshot()
        if plan.get('frame') != FRAME or plan.get('tcp') != TCP or not np.allclose(plan['offset_s'], np.arange(40)*.05, atol=1e-12, rtol=0):
            raise ValueError('Native reference frame/time mismatch')
        return gate_action(plan['rows10'], pose10(state['tcp'], state['gripper_fraction']),
                           state['time_s'], f"{self.card['id']}:{self.engine.steps}")

    def advance_atom(self, plan, point, *, raw=None):
        from ..task_data.codec import pose10
        if type(point) is not int or not 0 <= point < 8:
            raise ValueError('Only the executed prefix8 can become supervision')
        if self.forecast is None or self.forecast_index != point*25:
            raise PermissionError('Original S1 must admit each exact prefix before execution')
        before = self.snapshot()
        installed = pose10(plan['poses'][point], plan['opening'][point])
        if not np.array_equal(installed, plan['rows10'][point]):
            raise ValueError('Reference matrix and EEF10 disagree')
        memory = {k: deepcopy(getattr(self.engine, k)) for k in MEMORY}
        native_before = self._get_data(self.engine.data)
        consumed = []
        failure = None
        ticks0 = self.engine.steps
        for _ in range(25):
            t0 = float(self.engine.data.time)
            ticks_before = self.engine.steps
            try:
                self.engine.step(plan['poses'][point], float(plan['opening'][point]))
            except RuntimeError as error:
                failure = str(error)
            if self.engine.steps == ticks_before+1:
                consumed.append(dict(time_s=t0, eef10=installed.copy()))
                # Match the original high-rate record_tick observation, then
                # compare the live controller/integration state to original S1.
                self.snapshot()
                from scripts.pi0294.s1_preflight import observed_state
                expected = self.forecast[self.forecast_index]
                difference = float(np.max(np.abs(observed_state(self.engine)-expected)))
                self.max_shadow_error = max(self.max_shadow_error, difference)
                self.forecast_index += 1
                if difference > 1e-10:
                    raise RuntimeError('ORIGINAL_S1_LIVE_SHADOW_PARITY_FAILED')
            if failure:
                break
        after = self.snapshot()
        terminal = bool(failure or not after['safety_valid'])
        truncated = bool(not terminal and after['time_s'] >= 32.-1e-8)
        self.last_command = installed.copy()
        row = dict(episode_id=self.card['id'], scene_id=self.card['scene_group'],
            parent_id=self.card.get('parent_id', self.card['id']), split=self.card['split'],
            executor_id=self.executor_id, segment_id=self.segment_id, frame=FRAME, tcp=TCP,
            t0=before['time_s'], t1=after['time_s'], physics_ticks=self.engine.steps-ticks0,
            state_before=before, state_after=after, action_raw=deepcopy(raw),
            reference_installed=installed, reference_consumed=consumed, controller_memory=memory,
            controller_memory_after={k: deepcopy(getattr(self.engine, k)) for k in MEMORY},
            native_before=native_before, native_after=self._get_data(self.engine.data),
            context=dict(goal_xyz=[*self.card['goal_xy'], .46], table_top_m=.44, cube_edge_m=.04,
                         model_sha256=self.model_sha256),
            terminal=terminal, truncated=truncated, safety_reason=failure,
            planned_dt=.05, actual_dt=after['time_s']-before['time_s'],
            analysis_teacher_stage=self.teacher.stage)
        validate_atom(row)
        self.history.append(after)
        self.teacher.observe(after)
        return row

    def score(self, initial):
        from scripts.pi0294.score import score
        from ..target_sim.scorer import BOOL_FIELDS
        states = [initial]+self.history
        data = {key: np.asarray([s[key] for s in states], dtype=bool) for key in BOOL_FIELDS}
        data.update(time_s=np.asarray([s['time_s'] for s in states]),
            object_pose=np.asarray([s['object_pose'] for s in states]),
            tcp_world=np.asarray([s['tcp'] for s in states]),
            object_linear_velocity=np.asarray([s['object_linear_velocity'] for s in states]),
            object_angular_velocity=np.asarray([s['object_angular_velocity'] for s in states]))
        return score(data, self.card, complete=len(states) == 641)

    def close(self):
        self.engine.close()
