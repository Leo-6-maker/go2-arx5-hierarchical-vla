import copy
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model.close_reference_input_v1 import (
    TIME_METHOD, remember, query_reference, reference_arrays, prediction)
from go2arx5_vla.sla_adapter.resident_close_pair_v1 import shared_contract


def test_time_transport_uses_emitted_command_not_object_limited_measurement():
    adapter = SimpleNamespace(spec={'mobile_condition': {'method': TIME_METHOD}})
    row = np.zeros(10); row[9] = .012 / .044
    remember(adapter, row, {'source_native_s': 3.0}, {'step': 1525})
    sample = {'native_step': 1525, 'gripper_fraction': .9}
    context = query_reference(adapter, {'step': 1525, 'time_s': 3.05}, sample)
    assert context['width_m'] == .012
    data = dict(state_eef10=np.zeros(10), body_rgb=np.zeros((2, 2, 3)),
        wrist_rgb=np.ones((2, 2, 3)), T_policy_from_mount=np.eye(4), **reference_arrays(context))
    class FakeProvider:
        def predict(self, *args, **kwargs):
            return kwargs
    assert prediction(FakeProvider(), TIME_METHOD, data, 'pick', 7) == {
        'noise_id': 7, 'previous_reference': context}
    # Legacy providers receive exactly their old inputs, even if extra fields exist.
    assert prediction(FakeProvider(), 'MFREE_C20_FINAL_FM1000', data, 'pick', 7) == {'noise_id': 7}
    with pytest.raises(ValueError):
        query_reference(adapter, {'step': 1550, 'time_s': 3.10}, {'native_step': 1550})
    with pytest.raises(ValueError):
        query_reference(adapter, {'step': 1525, 'time_s': 3.10}, sample)
    data['previous_width_m'] = np.asarray([.012])
    with pytest.raises(ValueError):
        prediction(FakeProvider(), TIME_METHOD, data, 'pick', 7)


def test_resident_cannot_share_parent_across_changed_numeric_route_or_statistics():
    manifest = dict(bindings=dict(frozen_fingerprint='a', parent_host_cast_v7=True,
        parent_manifest={'sha256': 'b'}, tokenizer='c', inference_route='cached_actual',
        derived_statistics_sha256='d'), statistics_sha256='d', tokenizer_sha256='c',
        parent_manifest={'sha256': 'b'})
    shared_contract(manifest, copy.deepcopy(manifest))
    for key in ('frozen_fingerprint', 'parent_host_cast_v7', 'parent_manifest', 'tokenizer',
                'inference_route', 'derived_statistics_sha256'):
        changed = copy.deepcopy(manifest); changed['bindings'][key] = 'different'
        with pytest.raises(ValueError):
            shared_contract(manifest, changed)
