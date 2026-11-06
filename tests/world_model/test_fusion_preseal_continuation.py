"""Synthetic lifecycle seam only; no query, physics or native qualification."""
import copy
import json
import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_trajectory_qualification_native as native
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as contract
from test_fusion_trajectory_qualification import recipe, FakeAdapter, setup_flow
from test_fusion_reference_qualification_contract import reference

PHASE = 'collect-target-fusion-task-phase-live-skill-v8'


def prepare(recipe, tmp_path, monkeypatch, *, continuation=True):
    spec, _, _, episode, inputs = recipe
    spec = copy.deepcopy(spec)
    boundary = contract.endpoint(episode)
    if continuation:
        spec.update(phase=PHASE, kind='FUSION_TASK_PHASE_LIVE_SKILL_V8',
                    live_skill_start_step=boundary, max_ticks=boundary+50)

    class Adapter(FakeAdapter):
        def __init__(self):
            super().__init__(spec, tmp_path)
            self.trace_sealed = False
            self.order = []

        def install_reference(self, action):
            # The fixture checks the prefix recipe, while the new explicit
            # composite spec reserves a later final endpoint. No native grant.
            contract.validate_install_preview(action,
                initial_tcp=self.trajectory_initial_tcp, program=self.trajectory_program,
                step=self.step, base=self.base_pose_hold,
                frozen_base=self.trajectory_initial_base, end_step=boundary)
            self.active_reference = dict(reference(round(self.step*.002*1e9)/1e9),
                                         action_EEF10=action.tolist())
            self.installs.append((self.step, action.copy()))
            return self.active_reference

        def seal_trace(self, step):
            assert not self.trace_sealed
            self.order.append(('seal', step))
            self.trace_sealed = True
            super().seal_trace(step)

        def cancel_reference(self):
            self.order.append(('cancel', self.step))
            return super().cancel_reference()

    adapter = Adapter()
    setup_flow(monkeypatch, adapter)
    return adapter, episode, inputs, boundary


def synthetic_suffix(adapter, report, samples, windows):
    """Stand in for a separately tested consumer; this test has no physics."""
    assert not adapter.trace_sealed
    assert samples[-1]['native_step'] == windows[-1]['native_end_step']
    adapter.order.append(('continuation', adapter.step))
    complete = copy.deepcopy(report)
    start = adapter.step
    complete['windows'] += [dict(synthetic_suffix_only=True,
        native_start_step=start+25*i, native_end_step=start+25*(i+1)) for i in range(2)]
    total = len(complete['windows'])
    complete.update(actual_spec_phase=PHASE, native_endpoint_step=adapter.spec['max_ticks'],
        complete_task_atoms=total, solver_target_confirmed_atoms=total,
        gripper_confirmed_atoms=total)
    adapter.step = adapter.spec['max_ticks']
    return complete


def test_legacy_finalizer_remains_after_its_original_final_seal(recipe, tmp_path, monkeypatch):
    a, e, inputs, boundary = prepare(recipe, tmp_path, monkeypatch, continuation=False)
    def finalize(report):
        assert a.trace_sealed and a.sealed == boundary
        a.order.append(('finalizer', boundary))
    native.run_program(a, e, inputs, report_finalizer=finalize)
    assert a.order == [('cancel', boundary), ('seal', boundary), ('finalizer', boundary)]
    assert not (tmp_path/'LIVE_SKILL_INITIALIZATION_STAGE.private.json').exists()


def test_new_boundary_finalizer_and_saved_prefix_precede_query_then_only_final_seal(recipe, tmp_path, monkeypatch):
    a, e, inputs, boundary = prepare(recipe, tmp_path, monkeypatch)
    def finalize(report):
        assert not a.trace_sealed and a.step == boundary
        a.order.append(('finalizer', boundary))
        report['nine_sample_fixture_readback_done'] = True
    def continuation(adapter, report, samples, windows):
        assert json.loads((tmp_path/'LIVE_SKILL_INITIALIZATION_STAGE.private.json').read_text()) == report
        assert report['nine_sample_fixture_readback_done']
        assert report['operator_cancel']['kind'] == 'NO_CANCEL_AT_INTERMEDIATE_NATIVE_PAUSE'
        return synthetic_suffix(adapter, report, samples, windows)
    out = native.run_program(a, e, inputs, report_context=dict(actual_spec_phase=PHASE),
        report_finalizer=finalize, continuation=continuation)
    assert a.order == [('finalizer', boundary), ('continuation', boundary),
                       ('cancel', boundary+50), ('seal', boundary+50)]
    assert out['native_endpoint_step'] == boundary+50
    assert out['initialization_stage']['native_endpoint_step'] == boundary
    assert out['initialization_stage']['final_trace_sealed_at_this_boundary'] is False
    assert a.reference_qualification is out
    assert len(out['windows']) == (a.spec['max_ticks']-2503)//25


@pytest.mark.parametrize('fault', ['phase', 'boundary', 'endpoint', 'sealed'])
def test_new_continuation_requires_exact_future_phase_and_registered_distinct_clock_before_prepare(recipe, tmp_path, monkeypatch, fault):
    a, e, inputs, boundary = prepare(recipe, tmp_path, monkeypatch)
    if fault == 'phase': a.spec['phase'] = contract.PHASE
    elif fault == 'boundary': a.spec['live_skill_start_step'] += 25
    elif fault == 'endpoint': a.spec['max_ticks'] = boundary
    else: a.trace_sealed = True
    with pytest.raises(ValueError, match='EXPLICIT_V8_PREREGISTERED'):
        native.run_program(a, e, inputs, continuation=synthetic_suffix)
    assert not (tmp_path/'init.trace').exists()
    assert not a.installs


def test_failed_query_callback_keeps_unsealed_saved_prefix_for_original_outer_failure_cleanup(recipe, tmp_path, monkeypatch):
    a, e, inputs, boundary = prepare(recipe, tmp_path, monkeypatch)
    def failed(*args): raise TimeoutError('synthetic query failed; do not retry')
    with pytest.raises(TimeoutError): native.run_program(a, e, inputs, continuation=failed)
    assert a.step == boundary and not a.trace_sealed
    assert (tmp_path/'LIVE_SKILL_INITIALIZATION_STAGE.private.json').exists()
    assert not (tmp_path/'TRAJECTORY_QUALIFICATION.private.json').exists()


def test_continuation_cannot_rewrite_initialization_evidence(recipe, tmp_path, monkeypatch):
    a, e, inputs, boundary = prepare(recipe, tmp_path, monkeypatch)
    def altered(adapter, report, samples, windows):
        out = synthetic_suffix(adapter, report, samples, windows)
        report['complete_task_atoms'] = 0
        return out
    with pytest.raises(ValueError, match='UNCHANGED_PREFIX'):
        native.run_program(a, e, inputs, continuation=altered)
    assert not a.trace_sealed
