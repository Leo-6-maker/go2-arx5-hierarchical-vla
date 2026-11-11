import math
import numpy as np
import pytest

from go2arx5_vla.world_model.native_camera_transfer_v1 import (
    POSITION, LOOK_AT, camera_rotation, project_box, coverage, unchanged_packet)


def test_fixed_camera_follows_base_rotation_and_looks_at_base_point():
    r = camera_rotation()
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-15)
    assert np.linalg.det(r) == pytest.approx(1.)
    assert np.allclose(-r[:, 2], (LOOK_AT-POSITION)/np.linalg.norm(LOOK_AT-POSITION))
    # A nontrivial full-body rotation must rotate view/up as well as location.
    from scipy.spatial.transform import Rotation
    base = Rotation.from_euler('xyz', [.3, -.2, 1.]).as_matrix()
    assert np.allclose(base @ r[:, 2], (base @ r)[:, 2])
    assert not np.allclose((base @ r)[:, 1], r[:, 1])


def test_amodal_box_uses_occluded_corners_and_rejects_behind_camera():
    K = np.array([[100, 0, 128], [0, 100, 128], [0, 0, 1]])
    args = (np.eye(3), np.array([.1, .1, .1]), np.eye(4), K)
    assert project_box(np.array([0, 0, 1.]), *args)['margin_pass']
    assert not project_box(np.array([0, 0, .05]), *args)['all_in_front']
    result = project_box(np.array([1.1, 0, 1.]), *args)
    assert result['all_in_front'] and not result['margin_pass']
    assert result['bbox'][2] > 248


def test_coverage_has_full_population_and_phase_denominators():
    rows = []
    for i in range(1500):
        rows.append(dict(supervised=i<1450, phase=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')[i%5],
            body_projection=dict(margin_pass=i<1378), visible_pixels=dict(body_rgb=64, wrist_rgb=0)))
    result = coverage(rows)
    assert result['passes'] and result['amodal_required'] == 1378
    assert result['interventions'] == 50
    assert not coverage(rows, processed=True)['passes']
    rows[0]['body_projection']['margin_pass'] = False
    assert not coverage(rows)['passes']
    rows[0]['body_projection']['margin_pass'] = True
    for row in [r for r in rows if r['supervised'] and r['phase']=='CLOSE'][:30]:
        row['visible_pixels'] = dict(body_rgb=63, wrist_rgb=63)
    assert not coverage(rows)['phases']['CLOSE']['passes']
    assert not coverage(rows[:-1])['passes']


def test_reexport_cannot_change_labels_masks_or_wrist():
    packet = dict(actions=np.zeros((40, 10)), valid_window=np.ones(15, bool),
                  rgb_sha256=np.full((15, 2), 'a'*64), rgb_paths=np.full((15, 2), 'x.png'))
    out = {k:v.copy() for k,v in packet.items()}
    out['rgb_sha256'][:, 0] = 'b'*64
    unchanged_packet(packet, out)
    out['valid_window'][0] = False
    with pytest.raises(ValueError, match='NON_BODY_PACKET'):
        unchanged_packet(packet, out)
    out['valid_window'][0] = True
    out['rgb_sha256'][0, 1] = 'c'*64
    with pytest.raises(ValueError, match='NON_BODY_PACKET'):
        unchanged_packet(packet, out)
