"""Pure declaration of an explicit source-rig sensor on a target render copy.

This module reads no files and performs no rendering, FK, integration, model
query, or observation binding. Native camera records must first pass the old
same-packet camera_records validator. A future versioned capture/binder must
retain those records and identify the added rendering sensor separately.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

SCHEMA = 'WM02_FUSION_SOURCE_CAMERA_RIG_DECLARATION_V1'
RIG = 'SIM_EXTERNAL_WORLD_SOURCE_RIG'
OVERLAY_CAMERA = 'fusion_source_external_world_rgb_v1'
SOURCE_IDENTITY_SHA256 = '0fe9090e469e7b864213b23c3b98b82bd2bd0162a2c4afc4cea7820536091858'
SOURCE_OBSERVATION_SHA256 = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
SOURCE_XML_SHA256 = '7855eb0ab1af4677f4b9410b5971944e89195c0232f07dcc80c7af72edf2a703'
SOURCE_STATIC_SHA256 = 'b903afed7e875dfd4cdb08b192be2a3c81989155d3cb64cca20212c42403f23f'
SOURCE_CODE_SHA = '887e65629291f16f73863e2b111123007542d268'
SOURCE_CASE = 'PI0294-MP1-SOURCE-CONFIRM-S-CORR-2048-N29201-B0-F0-001-T1'
FRAME_MAP_SHA256 = 'fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1'
SOURCE_FRAME = 'PI0291_MUJOCO_WORLD'
OPTICAL_FROM_MUJOCO = ((1., 0., 0.), (0., -1., 0.), (0., 0., -1.))


def need(ok, reason):
    if not ok:
        raise ValueError('FUSION_SOURCE_CAMERA_RIG_' + reason)


def _json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'NONFINITE_JSON'))


def record_hash(value):
    """Canonical metadata hash, distinct from the binary rigid-map hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _numeric(value, shape):
    need(not any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat),
         'NUMERIC_ARRAY_NOT_BOOLEAN')
    result = np.asarray(value, dtype=float)
    need(result.shape == shape and np.isfinite(result).all(), 'FINITE_ARRAY_SHAPE')
    return result


def _rigid(value):
    result = _numeric(value, (4, 4))
    rotation = result[:3, :3]
    need(np.array_equal(result[3], [0, 0, 0, 1])
         and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
         and abs(np.linalg.det(rotation) - 1) <= 1e-10, 'RIGID_PROPER_TRANSFORM')
    return result


def _immutable(value):
    value = np.asarray(value, dtype='<f8')
    return np.frombuffer(value.tobytes(), dtype='<f8').reshape(value.shape)


def _intrinsic(value, fovy):
    need(type(fovy) in (int, float) and math.isfinite(fovy) and 0 < fovy < 180,
         'FINITE_PERSPECTIVE_FOV')
    focal = 128 / math.tan(math.radians(fovy) / 2)
    matrix = _numeric(value, (3, 3))
    need(np.allclose(matrix, [[focal, 0, 128], [0, focal, 128], [0, 0, 1]], rtol=0, atol=1e-9),
         'SAVED_256_INTRINSIC_AND_FOV')
    return matrix


def _xyaxes(value):
    axes = _numeric([float(x) for x in value.split()], (6,)).reshape(2, 3)
    x, y = axes
    need(abs(np.linalg.norm(x) - 1) < 1e-10 and abs(np.linalg.norm(y) - 1) < 1e-10
         and abs(x @ y) < 1e-10, 'SOURCE_XML_ORTHONORMAL_CAMERA_AXES')
    return np.column_stack((x, y, np.cross(x, y)))


def _quaternion(value):
    w, x, y, z = _numeric(value, (4,))
    need(abs(w*w + x*x + y*y + z*z - 1) < 1e-10, 'UNIT_NATIVE_CAMERA_QUATERNION')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass(frozen=True, init=False)
class SourceCameraRig:
    """A declaration only: external overlay plus the original native wrist.

The exact source audit recovers calibration from its saved same-pair static
qualification. T_source_from_target maps target-world points into source-world
coordinates; its inverse transports the fixed source external camera into the
target rendering world. The current target wrist keeps its actual extrinsics.
"""
    external_T_target_world_optical: np.ndarray = field(repr=False)
    external_T_target_world_mujoco: np.ndarray = field(repr=False)
    external_K: np.ndarray = field(repr=False)
    external_fovy_deg: float
    source_identity_audit_sha256: str
    native_camera_records_sha256: str
    _native_json: bytes = field(repr=False)
    _source_json: bytes = field(repr=False)

    def __init__(self, source_identity_bytes, T_source_from_target, native_camera_records,
                 *, expected_native_camera_records_sha256):
        need(type(source_identity_bytes) is bytes and 0 < len(source_identity_bytes) <= 2*1024**2
             and hashlib.sha256(source_identity_bytes).hexdigest() == SOURCE_IDENTITY_SHA256,
             'EXACT_SAVED_SOURCE_CAMERA_IDENTITY_AUDIT_BYTES')
        audit = _json(source_identity_bytes)
        need(audit.get('schema') == 'WM02_SOURCE71558_CAMERA_IDENTITY_CHAIN_RO_V1'
             and audit.get('status') == 'PASS_EXACT_71558_EPISODE_SCENE_CAMERA_CHAIN'
             and audit.get('errors') == [] and isinstance(audit.get('checks'), dict)
             and len(audit['checks']) >= 32 and all(v is True for v in audit['checks'].values())
             and audit.get('read_only') is True and audit.get('effects') == 0,
             'COMPLETE_SOURCE_IDENTITY_CHAIN')
        need(audit.get('source_observation_sha256') == SOURCE_OBSERVATION_SHA256
             and audit.get('source_XML_sha256') == SOURCE_XML_SHA256
             and audit.get('source_code_sha') == SOURCE_CODE_SHA and audit.get('source_episode_id') == SOURCE_CASE
             and audit.get('calibration_provenance') == 'PINNED_STATIC_QUALIFICATION_REUSED_SAME_INITIAL_RENDER_PAIR',
             'EXACT_SOURCE_OBSERVATION_SCENE_CODE_AND_SAMEPAIR_CALIBRATION')
        static_rows = [v for k, v in audit['files'].items()
                       if k.endswith('/episodes/'+SOURCE_CASE+'/STATIC_QUALIFICATION.json')]
        need(len(static_rows) == 1 and static_rows[0].get('sha256') == SOURCE_STATIC_SHA256,
             'PINNED_ORIGINAL_STATIC_QUALIFICATION_MEMBER')
        frame = _rigid(T_source_from_target)
        need(hashlib.sha256(frame.astype('<f8').tobytes()).hexdigest() == FRAME_MAP_SHA256,
             'EXACT_DECLARED_SOURCE_FROM_TARGET_FRAME_MAP')
        need(type(native_camera_records) is dict and set(native_camera_records) == {'body_rgb', 'wrist_rgb'}
             and record_hash(native_camera_records) == expected_native_camera_records_sha256,
             'PINNED_COMPLETE_ORIGINAL_NATIVE_CAMERA_RECORDS')
        native = _json(json.dumps(native_camera_records, allow_nan=False).encode())
        for name, item in native.items():
            need(item.get('name') == name and isinstance(item.get('parent_body'), str)
                 and bool(item['parent_body']) and type(item.get('width')) is int
                 and type(item.get('height')) is int and item['width'] == item['height'] == 256,
                 'ORIGINAL_NATIVE_NAMED_CAMERA_AND_RESOLUTION')
            _numeric(item['position_local_m'], (3,)); _quaternion(item['quaternion_local_wxyz'])
            _rigid(item['T_world_optical']); _intrinsic(item['K'], item['fovy_deg'])
        cameras, calibration = audit['camera_elements'], audit['source_calibration']
        need(set(cameras) == set(calibration) == {'body_rgb', 'wrist_rgb'}, 'EXACT_TWO_SOURCE_CAMERAS')
        for name, item in calibration.items():
            need(item.get('timestamp_s') == 0. and item.get('snapshot_id') == SOURCE_CASE+':0'
                 and item.get('parent') == ('world_fixed_rig' if name == 'body_rgb' else 'x5_link6'),
                 'SAME_INITIAL_SOURCE_RENDER_PAIR')
            _rigid(item['T_world_optical'])
            _intrinsic(item['K'], float(cameras[name]['attributes']['fovy']))
        body, wrist = cameras['body_rgb'], cameras['wrist_rgb']
        need(body['attributes']['name'] == 'body_rgb' and body['parent_chain'][0]['tag'] == 'worldbody'
             and wrist['attributes']['name'] == 'wrist_rgb'
             and wrist['parent_chain'][0]['attributes']['name'] == 'x5_link6', 'SOURCE_CAMERA_PARENT_IDENTITIES')
        external = _rigid(calibration['body_rgb']['T_world_optical'])
        expected_source = np.eye(4)
        expected_source[:3, :3] = _xyaxes(body['attributes']['xyaxes']) @ OPTICAL_FROM_MUJOCO
        expected_source[:3, 3] = _numeric([float(x) for x in body['attributes']['pos'].split()], (3,))
        need(np.allclose(external, expected_source, rtol=0, atol=1e-9)
             and float(body['attributes']['fovy']) == 55., 'ACTUAL_FIXED_SOURCE_EXTERNAL_WORLD_RIG')
        nw = native['wrist_rgb']
        source_wrist_pos = _numeric([float(x) for x in wrist['attributes']['pos'].split()], (3,))
        need(nw['parent_body'] == 'x5_link6' and nw['fovy_deg'] == float(wrist['attributes']['fovy']) == 75.
             and np.allclose(nw['position_local_m'], source_wrist_pos, rtol=0, atol=1e-12)
             and np.allclose(_quaternion(nw['quaternion_local_wxyz']), _xyaxes(wrist['attributes']['xyaxes']),
                             rtol=0, atol=1e-10), 'ORIGINAL_NATIVE_WRIST_LOCAL_GEOMETRY_PARITY_REQUIRED')
        overlay = _rigid(np.linalg.inv(frame) @ external)
        mujoco = overlay.copy(); mujoco[:3, :3] = overlay[:3, :3] @ OPTICAL_FROM_MUJOCO
        object.__setattr__(self, 'external_T_target_world_optical', _immutable(overlay))
        object.__setattr__(self, 'external_T_target_world_mujoco', _immutable(mujoco))
        object.__setattr__(self, 'external_K', _immutable(calibration['body_rgb']['K']))
        object.__setattr__(self, 'external_fovy_deg', 55.)
        object.__setattr__(self, 'source_identity_audit_sha256', hashlib.sha256(source_identity_bytes).hexdigest())
        object.__setattr__(self, 'native_camera_records_sha256', expected_native_camera_records_sha256)
        object.__setattr__(self, '_native_json', json.dumps(native, allow_nan=False).encode())
        geometry_keys = ('K', 'T_world_optical', 'parent', 'snapshot_id', 'timestamp_s', 'optical_axis_convention')
        geometry = {name: {key: value[key] for key in geometry_keys if key in value}
                    for name, value in calibration.items()}
        object.__setattr__(self, '_source_json', json.dumps(dict(cameras=cameras, calibration=geometry), allow_nan=False).encode())

    def declaration(self):
        """JSON-safe proposed sensor schema; this is not a captured observation."""
        native = _json(self._native_json); source = _json(self._source_json)
        return dict(schema=SCHEMA, rig=RIG, status='DECLARATION_ONLY_NOT_RENDERED',
            source_identity_audit_sha256=self.source_identity_audit_sha256,
            source_observation_sha256=SOURCE_OBSERVATION_SHA256, source_XML_sha256=SOURCE_XML_SHA256,
            frame_map_sha256=FRAME_MAP_SHA256, source_world_frame=SOURCE_FRAME,
            frame_rule='T_TARGET_WORLD_OPTICAL=INVERSE_T_SOURCE_FROM_TARGET@T_SOURCE_WORLD_OPTICAL',
            native_target_camera_records=native, native_camera_records_sha256=self.native_camera_records_sha256,
            source_saved_camera_provenance=source,
            policy_camera_role_mapping={'external_rgb': OVERLAY_CAMERA, 'wrist_rgb': 'wrist_rgb'},
            declared_rendering_sensors={
                OVERLAY_CAMERA: dict(name=OVERLAY_CAMERA, origin='DECLARED_SOURCE_EXTERNAL_WORLD_RIG_ON_RENDER_COPY',
                    parent='TARGET_RENDER_WORLD_FIXED', T_world_optical=self.external_T_target_world_optical.tolist(),
                    T_world_mujoco=self.external_T_target_world_mujoco.tolist(), K=self.external_K.tolist(),
                    width=256, height=256, fovy_deg=self.external_fovy_deg),
                'wrist_rgb': dict(name='wrist_rgb', origin='ORIGINAL_NATIVE_WRIST_SAME_PACKET_CAMERA',
                    local_geometry_matches_source=True, T_world_optical=native['wrist_rgb']['T_world_optical'],
                    K=native['wrist_rgb']['K'], width=256, height=256, fovy_deg=75.)},
            requirements=dict(independent_CPU_render_copy=True, original_native_MJB_unchanged=True,
                original_native_camera_records_retained=True, original_same_packet_native_FK_validation=True,
                new_versioned_overlay_binding=True, actual_overlay_PNG_and_calibration_hashes=True,
                actual_same_snapshot_packet_and_clock=True, actual_CPU_renderer_proof=True,
                source_PNG_replacement_forbidden=True),
            effects=dict(renders=0, FK_calls=0, integrations=0, model_queries=0, files_written=0),
            native_camera_equivalence=False, image_domain_equivalence=False, occlusion_equivalence=False,
            target_embodiment_qualified=False, target_task_qualified=False, native_execution_authorized=False,
            source_images_supplied=False, observation_payload_created=False)

    def render_copy_patch(self, slot='body_rgb'):
        """Describe replacement of one copy-only camera slot; mutate no model.

        Keeping the model's existing slot name does not turn the declared world
        rig into the original native body camera. Both identities must be saved
        by the future capture/binder, together with the actual rendered bytes.
        """
        need(slot == 'body_rgb', 'ONLY_DECLARED_EXTERNAL_RENDER_COPY_SLOT')
        return dict(schema='WM02_FUSION_SOURCE_CAMERA_RENDER_COPY_PATCH_V1',
            existing_render_copy_slot=slot, declared_sensor_name=OVERLAY_CAMERA,
            rig=RIG, parent_body_id=0, parent='TARGET_RENDER_WORLD_FIXED',
            position_m=self.external_T_target_world_mujoco[:3, 3].tolist(),
            rotation_matrix_mujoco=self.external_T_target_world_mujoco[:3, :3].tolist(),
            fovy_deg=self.external_fovy_deg, width=256, height=256,
            independent_render_model_copy_required=True, original_native_model_write_authorized=False,
            source_identity_audit_sha256=self.source_identity_audit_sha256,
            native_camera_records_sha256=self.native_camera_records_sha256,
            frame_map_sha256=FRAME_MAP_SHA256, status='PATCH_DESCRIPTOR_ONLY_NOT_APPLIED')
