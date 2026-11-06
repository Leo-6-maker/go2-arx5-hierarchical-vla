"""Hash-bound target camera/state input for frozen-source VLA shadow only.

Consumes the existing FrameRecorder's bytes and ActualState reader. It never
renders, integrates, loads Torch, installs references or supplies source outcomes.
The caller must independently verify the scene bytes and actual capture/readback;
hashes and CPU fixtures alone cannot establish that rendering actually occurred.
"""
from dataclasses import dataclass
import base64
import copy
import hashlib
import io
import json

import numpy as np
from PIL import Image

from scripts.pi0292.eval_contract import FRAME, TCP, matrix10
from scripts.pi0292.relay_protocol import sealed, unseal, uuid_string, validate_observation
from scripts.pi0294.dec1.target_observation import rigid, transform_hash
from scripts.pi0294.task_state import ActualState, parse_state
from ..trajectory.so3 import rotation_matrix_to_rot6d_rows

SCHEMA = 'WM02_FUSION_TARGET_OBSERVATION_SHADOW_V1'
CAMERAS = ('body_rgb', 'wrist_rgb')
ROLES = {'external_rgb': 'body_rgb', 'wrist_rgb': 'wrist_rgb'}


def require(ok, why):
    if not ok:
        raise ValueError('FUSION_OBSERVATION_' + why)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_bytes(raw, expected, name, maximum=4 * 1024**2):
    require(type(raw) is bytes and 0 < len(raw) <= maximum, name + '_BOUNDED_BYTES')
    require(type(expected) is str and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected)
            and sha256(raw) == expected, name + '_BYTES_CHANGED')


def finite_number(value):
    return type(value) in (int, float) and np.isfinite(value)


def json_object(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'DUPLICATE_CALIBRATION_FIELD')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: require(False, 'FINITE_CALIBRATION'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('FUSION_OBSERVATION_CALIBRATION_JSON') from error
    require(type(value) is dict, 'CALIBRATION_OBJECT')
    return value


def camera_records(actual_state, calibration, stamp):
    """Check calibration against the same measured FK copy; keep actual rigs."""
    require(set(calibration) == set(CAMERAS), 'TWO_ACTUAL_CAMERA_CALIBRATIONS')
    model, data = actual_state.model, actual_state.data
    rows = {}
    for name in CAMERAS:
        cid = int(model.camera(name).id)
        item = calibration[name]
        require(type(item) is dict and set(item) == {'timestamp_s', 'T_world_optical', 'K'}
                and finite_number(item['timestamp_s']) and item['timestamp_s'] == stamp,
                'SAME_ABSOLUTE_CAMERA_CLOCK')
        optical = rigid(item['T_world_optical'])
        actual = np.eye(4)
        actual[:3, :3] = np.asarray(data.cam_xmat[cid]).reshape(3, 3) @ np.diag([1., -1., -1.])
        actual[:3, 3] = data.cam_xpos[cid]
        require(np.allclose(optical, rigid(actual), rtol=0, atol=1e-9),
                'CALIBRATION_DIFFERS_FROM_SAME_NATIVE_PACKET_FK')
        fovy = float(model.cam_fovy[cid])
        require(np.isfinite(fovy) and 0 < fovy < 180, 'ACTUAL_PERSPECTIVE_FOV')
        focal = 128 / np.tan(np.deg2rad(fovy) / 2)
        require(not any(isinstance(v, (bool, np.bool_)) for v in np.asarray(item['K'], dtype=object).flat),
                'NUMERIC_INTRINSICS')
        intrinsic = np.asarray(item['K'], float)
        require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all()
                and np.allclose(intrinsic, [[focal, 0, 128], [0, focal, 128], [0, 0, 1]],
                                rtol=0, atol=1e-9), 'ORIGINAL_256_PIXEL_INTRINSICS')
        parent = int(model.cam_bodyid[cid])
        rows[name] = dict(name=name, parent_body=str(model.body(parent).name),
            position_local_m=np.asarray(model.cam_pos[cid], float).tolist(),
            quaternion_local_wxyz=np.asarray(model.cam_quat[cid], float).tolist(),
            fovy_deg=fovy, width=256, height=256,
            T_world_optical=optical.tolist(), K=intrinsic.tolist())
    require(int(model.camera(CAMERAS[0]).id) != int(model.camera(CAMERAS[1]).id),
            'DISTINCT_ACTUAL_CAMERAS')
    return rows


@dataclass(frozen=True)
class TargetShadowBinding:
    """One cold, explicit frame relation; None retains target input without a view.

    T_source_from_target maps target-world TCP into the source policy world. This
    is an owner-chosen rigid coordinate relation, never a camera-equivalence or
    behavior/outcome calibration. Category 10 continues to mean the frozen source
    policy; it is not a new qualified Go2+ARX5 embodiment.
    """
    target_world_frame: str
    scene_sha256: str
    target_runtime_sha256: str
    T_source_from_target: object = None
    expected_transform_sha256: str = None

    def __post_init__(self):
        require(type(self.target_world_frame) is str and bool(self.target_world_frame)
                and self.target_world_frame != FRAME, 'EXPLICIT_ACTUAL_TARGET_WORLD_FRAME')
        for name in ('scene_sha256', 'target_runtime_sha256'):
            value = getattr(self, name)
            require(type(value) is str and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value), 'PINNED_' + name.upper())
        if self.T_source_from_target is None:
            require(self.expected_transform_sha256 is None, 'NO_HASH_WITHOUT_FRAME_RELATION')
        else:
            transform = rigid(self.T_source_from_target)
            require(transform_hash(transform) == self.expected_transform_sha256,
                    'PINNED_SOURCE_FROM_TARGET_TRANSFORM')
            transform.setflags(write=False)
            object.__setattr__(self, 'T_source_from_target', transform)

    def bind(self, *, actual_state, native_state_bytes, expected_state_sha256,
             capture, calibration_bytes, expected_calibration_sha256,
             image_bytes, expected_image_sha256, language, snapshot_uuid):
        """Bind already captured bytes; FrameRecorder.save supplies capture.

        Required capture fields are index/native_step/time_s, unchanged from the
        original recorder. All expected hashes must be pinned by the owning
        capture manifest. There is no legacy 2500+200*n query-step restriction.
        """
        require(isinstance(actual_state, ActualState), 'ORIGINAL_ACTUAL_STATE_READER')
        checked_bytes(native_state_bytes, expected_state_sha256, 'NATIVE_STATE', 2 * 1024**2)
        checked_bytes(calibration_bytes, expected_calibration_sha256, 'CALIBRATION', 2 * 1024**2)
        packet = parse_state(native_state_bytes)
        step, stamp = packet['step'], packet['time_s']
        require(finite_number(stamp) and stamp >= 0 and abs(stamp - step * .002) <= 1e-8,
                'ORIGINAL_ABSOLUTE_NATIVE_CLOCK')
        require(type(capture) is dict and set(capture) == {'index', 'native_step', 'time_s'}
                and type(capture['index']) is int and capture['index'] >= 0
                and type(capture['native_step']) is int and capture['native_step'] == step
                and finite_number(capture['time_s']) and capture['time_s'] == stamp,
                'SAME_SNAPSHOT_FRAME_RECORDER_CAPTURE')
        require(type(image_bytes) is dict and set(image_bytes) == set(CAMERAS)
                and type(expected_image_sha256) is dict and set(expected_image_sha256) == set(CAMERAS),
                'NAMED_TARGET_CAMERA_BYTES_AND_HASHES')
        for name in CAMERAS:
            checked_bytes(image_bytes[name], expected_image_sha256[name], name.upper())
            with Image.open(io.BytesIO(image_bytes[name])) as picture:
                require(picture.format == 'PNG' and picture.mode == 'RGB' and picture.size == (256, 256),
                        'ORIGINAL_TARGET_256_RGB_PNG')
                picture.verify()
        uuid_string(snapshot_uuid)
        require(type(language) is str and 0 < len(language) <= 4096, 'ORIGINAL_TASK_LANGUAGE')
        for contact in packet['contacts']:
            require(all(np.isfinite(np.asarray(contact[key])).all() for key in
                        ('distance_m', 'wrench', 'normal', 'frame', 'point')), 'FINITE_NATIVE_CONTACTS')
        # ActualState.update performs mj_forward on its own copy, never mj_step.
        measured = actual_state.update(packet)
        require(measured['native_step'] == step and measured['time_s'] == stamp
                and measured['safety_valid'] is True and measured['gripper_state_valid'] is True
                and measured['native_abort'] is False, 'SAME_VALID_MEASURED_NATIVE_FEEDBACK')
        tcp = rigid(measured['tcp'])
        grip = measured['gripper_fraction']
        require(finite_number(grip) and 0 <= grip <= 1, 'MEASURED_MECHANICAL_GRIPPER_NO_CLIP')
        calibration = json_object(calibration_bytes)
        cameras = camera_records(actual_state, calibration, stamp)
        state10 = np.r_[tcp[:3, 3], rotation_matrix_to_rot6d_rows(tcp[:3, :3]), grip]
        policy = None
        if self.T_source_from_target is not None:
            # Frozen dataclass/read-only NumPy flags are not a substitute for a
            # fresh content check if a caller has retained and altered an array.
            require(transform_hash(self.T_source_from_target) == self.expected_transform_sha256,
                    'FRAME_RELATION_CHANGED')
            source_tcp = rigid(self.T_source_from_target @ tcp)
            source_row = np.r_[source_tcp[:3, 3], rotation_matrix_to_rot6d_rows(source_tcp[:3, :3]), grip]
            images = [dict(camera=slot, png_base64=base64.b64encode(image_bytes[actual]).decode(),
                           sha256=expected_image_sha256[actual]) for slot, actual in ROLES.items()]
            payload = dict(snapshot_uuid=snapshot_uuid, language=language, state_eef10=source_row.tolist(),
                           images=images, frame=FRAME, tcp=TCP)
            validate_observation(payload)
            policy = dict(payload=payload, relation='EXPLICIT_MAPPED_POLICY_FRAME_VIEW',
                          T_source_from_target=self.T_source_from_target.tolist(),
                          transform_sha256=self.expected_transform_sha256)
        return sealed(dict(schema=SCHEMA, native_step=step, native_time_s=stamp,
            snapshot_uuid=snapshot_uuid, capture=copy.deepcopy(capture),
            target_observation=dict(frame=self.target_world_frame, tcp=TCP,
                actual_state_eef10=state10.tolist(), cameras=cameras,
                image_sha256=copy.deepcopy(expected_image_sha256), native_state_sha256=expected_state_sha256,
                calibration_sha256=expected_calibration_sha256, scene_sha256=self.scene_sha256,
                target_runtime_sha256=self.target_runtime_sha256),
            source_policy_view=policy,
            provenance=dict(camera_role_mapping=dict(ROLES), camera_equivalence=False,
                source_camera_domain_shift='UNQUALIFIED_SOURCE_TRAINED_POLICY_ON_ACTUAL_TARGET_VIEWS',
                frame_relation='EXPLICIT_RIGID_MAP' if policy is not None else 'SEMANTICS_UNRESOLVED',
                desired_outcome_provenance='SEMANTICS_UNRESOLVED', source_outcomes_supplied=False,
                processor='UNCHANGED_FROZEN_SOURCE_PROCESSOR', embodiment='new_embodiment', embodiment_id=10,
                embodiment_meaning='SOURCE_FIXED_BASE_ARX5_POLICY_EVALUATED_ON_TARGET_INPUT',
                target_embodiment_qualified=False, target_task_native_qualified=False,
                shadow_only=True, native_execution_authorized=False, online_closed_loop=False,
                clock='ORIGINAL_ABSOLUTE_TARGET_SIM_S_NO_RETIMING',
                raw_image_and_state_retention='OWNER_REQUIRED_BINDING_DOES_NOT_WRITE',
                capture_actuality='OWNER_CAPTURE_AND_ORIGINAL_BYTE_READBACK_REQUIRED',
                scene_identity='OWNER_VERIFIED_COMPILED_SCENE_BYTES_REQUIRED',
                source_H40_command_offsets_s=(np.arange(40) * .05).tolist(),
                outcome_times_are_not_command_issue_times=True)))


def shadow_inference_payload(bound):
    """Read the mapped payload only from the complete hash-bound shadow record."""
    body = unseal(bound)
    require(body.get('schema') == SCHEMA, 'VERSIONED_SHADOW_RECORD')
    provenance = body['provenance']
    require(provenance['shadow_only'] is True and provenance['native_execution_authorized'] is False
            and provenance['target_embodiment_qualified'] is False
            and provenance['desired_outcome_provenance'] == 'SEMANTICS_UNRESOLVED'
            and provenance['camera_role_mapping'] == ROLES and provenance['camera_equivalence'] is False,
            'TARGET_PROVENANCE_SIDECAR_REQUIRED')
    view = body['source_policy_view']
    if view is None:
        raise PermissionError('FUSION_OBSERVATION_UNRESOLVED_FRAME_RELATION_NO_SOURCE_POLICY_VIEW')
    require(view['relation'] == 'EXPLICIT_MAPPED_POLICY_FRAME_VIEW'
            and transform_hash(view['T_source_from_target']) == view['transform_sha256'], 'EXPLICIT_POLICY_FRAME_MAP')
    validate_observation(view['payload'])
    target, payload = body['target_observation'], view['payload']
    require(provenance['frame_relation'] == 'EXPLICIT_RIGID_MAP'
            and target['frame'] != FRAME and target['tcp'] == TCP
            and payload['snapshot_uuid'] == body['snapshot_uuid'], 'PRESERVED_TARGET_FRAME_AND_SNAPSHOT')
    target_state = np.asarray(target['actual_state_eef10'], float)
    require(target_state.shape == (10,) and np.isfinite(target_state).all(), 'PRESERVED_TARGET_STATE')
    mapped = rigid(view['T_source_from_target']) @ matrix10(target_state)
    expected = np.r_[mapped[:3, 3], rotation_matrix_to_rot6d_rows(mapped[:3, :3]), target_state[9]]
    require(np.allclose(payload['state_eef10'], expected, rtol=0, atol=1e-9), 'POLICY_STATE_DIFFERS_FROM_TARGET_FRAME_MAP')
    require(all(image['sha256'] == target['image_sha256'][ROLES[image['camera']]]
                for image in payload['images']), 'POLICY_IMAGES_DIFFER_FROM_TARGET_CAPTURE')
    return copy.deepcopy(view['payload'])
