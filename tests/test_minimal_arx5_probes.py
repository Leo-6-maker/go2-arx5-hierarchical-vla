from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_probes as p
from scripts.pi0294.generalization import minimal_study as study


def fixture_capture():
    root = Path(__file__).resolve().parents[1]/'reports/PI0294/design/parallel-science-20261022'
    panel = json.loads((root/'ARX5_MINIMAL_ADAPTATION_TWELVE_PARENT_AND_CURSOR_METADATA_FROZEN_20261027.json').read_text())['panel']
    images = [dict(camera=c, sha256=hashlib.sha256(c.encode()).hexdigest()) for c in ['external_rgb', 'wrist_rgb']]
    obs = []
    for i in range(0, 12, 2):
        a, b = panel['parents'][i:i+2]
        obs.append(dict(group=a['group'], parent_T1=a['id'], parent_T2=b['id'], anchor=16, stage='approach',
                        first_contact_anchor=29, split='TRAIN', new_labels=False, state_eef10=[0.] * 10,
                        images=deepcopy(images), instructions={'T1': 'CPU_FIXTURE_PICK', 'T2': 'CPU_FIXTURE_PUSH'},
                        source_numeric_sha256={'T1': a['sha256'], 'T2': b['sha256']}))
    return panel, dict(status='ACTUAL_EXISTING_SIX_TRAIN_APPROACH_OBSERVATIONS_NO_MODEL_OR_RENDER',
                      observations=obs, new_scientific_queries=0, new_physics=0, new_capture=0)


def test_small_panel_has_same_state_RGB_noise_pairs_and_no_counterfactual_BC_labels():
    panel, capture = fixture_capture(); out = p.prepare(capture, panel)
    assert len(out['cases']) == 24 and out['three_model_scientific_generation_upper_bound'] == 72
    for i in range(0, 24, 4):
        a, b, c, d = out['cases'][i:i+4]
        assert a['noise_seed'] == c['noise_seed'] and b['noise_seed'] == d['noise_seed']
        assert a['semantic_input_sha256'] != c['semantic_input_sha256']
        assert len({x['same_RGB_state_without_language_sha256'] for x in [a, b, c, d]}) == 1
    assert all(not r['usable_as_BC_training_label'] for r in out['cases'])
    assert out['scientific_queries_executed'] == 0 and out['native_adopted'] is False


@pytest.mark.parametrize('field,value', [('split', 'DEV'), ('stage', 'transport'), ('anchor', 50),
                                      ('first_contact_anchor', 15), ('new_labels', True)])
def test_grasped_or_post_contact_or_DEV_or_new_label_inputs_are_rejected(field, value):
    panel, capture = fixture_capture(); capture['observations'][0][field] = value
    with pytest.raises(ValueError, match='PRECONTACT'):
        p.prepare(capture, panel)


def test_wire_input_hash_excludes_UUID_but_checks_actual_PNG_state_and_instruction():
    image = b'CPU_FIXTURE_PNG_BYTES_NOT_A_RENDER'
    row = lambda name: dict(camera=name, sha256=hashlib.sha256(image).hexdigest(), png_base64=base64.b64encode(image).decode())
    payload = dict(snapshot_uuid='CPU_FIXTURE_UUID', state_eef10=[0.] * 10,
                   images=[row('external_rgb'), row('wrist_rgb')], language='pick', frame=p.FRAME, tcp=p.TCP)
    first = p.wire_input_fingerprint(payload)
    payload['snapshot_uuid'] = 'DIFFERENT_UUID'
    assert p.wire_input_fingerprint(payload) == first
    payload['language'] = 'push'
    assert p.wire_input_fingerprint(payload) != first
    payload['images'][0]['png_base64'] = base64.b64encode(b'CHANGED').decode()
    with pytest.raises(ValueError, match='PNG_BYTES_CHANGED'):
        p.wire_input_fingerprint(payload)


def test_existing_wire_payload_preserves_real_observation_and_never_sends_teacher_target(tmp_path):
    import numpy as np
    panel, capture = fixture_capture()
    obs = capture['observations'][0]
    for row in obs['images']:
        f = tmp_path/(row['camera'] + '.png'); data = row['camera'].encode(); f.write_bytes(data)
        row['path'] = '/OLD_OTHER_HOST_NOT_MOUNTED/'+f.name
    prepared = p.prepare(capture, panel)
    case = prepared['cases'][2]  # Same T1 state, the paired T2 instruction.
    numeric=tmp_path/'numeric.npz'
    np.savez(numeric,rgb_paths=np.array([[x['camera']+'.png'for x in obs['images']]]*17),
        rgb_sha256=np.array([[x['sha256']for x in obs['images']]]*17))
    digest=hashlib.sha256(numeric.read_bytes()).hexdigest()
    # Deliberately tiny numeric fixture, not a frozen production parent.
    obs['source_numeric_sha256']['T1']=digest
    class Reader:
        root=tmp_path
        raw_bindings={case['source_parent']:str(tmp_path)}
        rows = [({'id': case['source_parent'],'file':'numeric.npz','sha256':digest,'raw_root':'/OLD_HOST'}, 16)]
        def raw(self, index):
            return obs['state_eef10'], 'TEACHER_ARRAY_MUST_NOT_BE_SENT', None, obs['instructions']['T1']
    payload = p.payload_for_case(case, capture, Reader())
    assert set(payload) == {'snapshot_uuid', 'state_eef10', 'images', 'language', 'frame', 'tcp'}
    assert payload['language'] == obs['instructions']['T2']
    assert p.wire_input_fingerprint(payload) == case['semantic_input_sha256']
    # An old capture path being unreadable was irrelevant; changing the real
    # selected host's PNG still fails, without recompressing or substituting it.
    (tmp_path/'external_rgb.png').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='PNG_BYTES_CHANGED'):
        p.payload_for_case(case,capture,Reader())


@pytest.mark.parametrize('escape',['../outside.png','/outside.png'])
def test_native_language_payload_rejects_numeric_image_escape(tmp_path,escape):
    import numpy as np
    from types import SimpleNamespace
    h=hashlib.sha256(b'CPU_PNG').hexdigest();numeric=tmp_path/'n.npz'
    np.savez(numeric,rgb_paths=np.array([[escape,'wrist.png']]*17),rgb_sha256=np.array([[h,h]]*17))
    digest=hashlib.sha256(numeric.read_bytes()).hexdigest()
    obs=dict(anchor=16,parent_T1='CPU_PARENT',source_numeric_sha256=dict(T1=digest),
        images=[dict(camera=x,sha256=h)for x in ['external_rgb','wrist_rgb']])
    reader=SimpleNamespace(root=tmp_path,raw_bindings={'CPU_PARENT':str(tmp_path)},
        rows=[(dict(id='CPU_PARENT',file='n.npz',sha256=digest,raw_root=str(tmp_path)),16)])
    with pytest.raises(ValueError,match='ORIGINAL_TWO_PNG_PATHS'):
        p.bound_reader_images(obs,reader,0)


def test_actual_precontact_capture_freezes_all24_without_neural_or_physics(tmp_path):
    name = os.environ.get('ARX5_PRECONTACT_CAPTURE')
    if not name:
        pytest.skip('Private actual source readback not provided')
    panel, _ = fixture_capture()
    transport = json.loads(Path(name).read_text()); assert transport['returncode'] == 0
    capture = json.loads(transport['stdout']); result = p.prepare(capture, panel)
    assert len(result['cases']) == 24 and result['scientific_queries_executed'] == 0
    assert all(r['anchor'] == 16 and not r['usable_as_BC_training_label'] for r in result['cases'])
