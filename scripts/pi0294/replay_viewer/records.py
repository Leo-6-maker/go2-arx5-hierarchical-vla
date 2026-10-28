"""Read original PI0294 records without relabelling, interpolation or rescoring.

Fixed-base fields follow episode.py; native-body fields follow task_state.py and
night_r1_worker.py. All paths remain local. Missing fields stay missing.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

CAMERAS = ('body_rgb', 'wrist_rgb')
KINDS = ('teacher', 'student_eval', 'offline_prediction')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('ORIGINAL_JSON_SHA256_MISMATCH')
    return json.loads(raw), digest


def local_file(root, name):
    p = root / name
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('LOCAL_RELATIVE_PATH_REQUIRED')
    if not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('SOURCE_FILE_OUTSIDE_EPISODE_OR_MISSING')
    return p


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('NONFINITE_RECORDED_NUMBER')
    return value


def pose(value):
    if value is None:
        return None
    if not (isinstance(value, list) and len(value) == 4 and all(isinstance(r, list) and len(r) == 4 for r in value)):
        raise ValueError('RECORDED_POSE_MUST_BE_4X4')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for row in value for x in row):
        raise ValueError('RECORDED_POSE_NUMERIC_REQUIRED')
    return finite(value)


class Episode:
    def __init__(self, spec):
        self.spec = dict(spec)
        self.id = spec['id']
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in self.id):
            raise ValueError('EPISODE_ID')
        self.kind = spec['source_kind']
        if self.kind not in KINDS:
            raise ValueError('EXPLICIT_SOURCE_KIND_REQUIRED')
        self.root = Path(spec['root']).resolve(strict=True)
        self.fixture = spec.get('fixture', False)
        if not isinstance(self.fixture, bool):
            raise ValueError('FIXTURE_BOOLEAN_REQUIRED')
        # A viewer does not infer success from positions or from completion status.
        ap = local_file(self.root, spec['audit_path'])
        try:
            self.audit, _ = json_bytes(ap, spec['audit_sha256'])
        except ValueError as exc:
            raise ValueError('ORIGINAL_AUDIT_SHA256_MISMATCH') from exc
        fmt = spec['format']
        sample_name = spec.get('samples_path', 'samples.json' if fmt == 'fixed_base' else 'samples.jsonl')
        if fmt not in ('fixed_base', 'native_body'):
            raise ValueError('UNKNOWN_RECORDED_FORMAT')
        sp = local_file(self.root, sample_name)
        sample_raw = sp.read_bytes()
        self.samples_sha256 = hashlib.sha256(sample_raw).hexdigest()
        if self.samples_sha256 != spec['samples_sha256']:
            raise ValueError('ORIGINAL_SAMPLES_SHA256_MISMATCH')
        rows = json.loads(sample_raw) if fmt == 'fixed_base' else [json.loads(s) for s in sample_raw.decode().splitlines() if s.strip()]
        if not isinstance(rows, list) or not rows:
            raise ValueError('NO_SAVED_OBSERVATION_FRAMES')
        self.view = spec.get('recorded_view', {'kind': 'full_saved_observations'})
        self.source_indices = list(range(len(rows)))
        if self.view['kind'] == 'recorded_keyframe_subset':
            indices = self.view['source_frame_indices']
            total = self.view['source_frame_count']
            if (len(indices) != len(rows) or not isinstance(total, int) or total < len(rows)
                    or any(not isinstance(i, int) or i < 0 or i >= total for i in indices)
                    or any(b <= a for a, b in zip(indices, indices[1:]))):
                raise ValueError('EXPLICIT_SOURCE_KEYFRAME_INDICES_REQUIRED')
            receipt, _ = json_bytes(local_file(self.root, self.view['selection_receipt']['path']), self.view['selection_receipt']['sha256'])
            row_hashes = [hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest() for r in rows]
            if (receipt['source_frame_indices'] != indices or receipt['source_frame_count'] != total
                    or receipt['original_samples_sha256'] != self.view['original_samples_sha256']
                    or receipt['selected_row_canonical_sha256'] != row_hashes):
                raise ValueError('DERIVED_KEYFRAME_SELECTION_IDENTITY')
            self.source_indices = indices
        elif self.view['kind'] != 'full_saved_observations':
            raise ValueError('UNKNOWN_SAVED_VIEW_KIND')
        self.image_roots = {'episode': self.root}
        for key, value in spec.get('image_roots', {}).items():
            if key == 'episode':
                raise ValueError('CANNOT_OVERRIDE_EPISODE_IMAGE_ROOT')
            self.image_roots[key] = Path(value).resolve(strict=True)
        self.frames = []
        self.assets = {}
        self._highrate = None
        if (self.root / 'highrate.npz').exists() and 'highrate_sha256' in spec:
            self._load_highrate(spec['highrate_sha256'])
        for i, row in enumerate(rows):
            self.frames.append(self._frame(row, i, fmt))
        times = [f['time_s'] for f in self.frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('SAVED_FRAME_CLOCK_MUST_INCREASE')
        self.events = []
        for name in ('TEACHER_EVENTS.json', 'S1_PREFLIGHTS.json'):
            if (self.root / name).exists():
                p = local_file(self.root, name)
                event_rows, digest = json_bytes(p)
                self.events.append({'file': name, 'sha256': digest, 'original_records': finite(event_rows)})

    def _load_highrate(self, expected):
        import numpy as np
        p = local_file(self.root, 'highrate.npz')
        from io import BytesIO
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('HIGH_RATE_IDENTITY_CHANGED')
        with np.load(BytesIO(raw), allow_pickle=False) as z:
            rows = z['rows'].copy()
            nq, nv = int(z['nq']), int(z['nv'])
        nu = rows.shape[1] - 1 - nq - nv - 7 if rows.ndim == 2 else -1
        if nq <= 0 or nv <= 0 or nu < 0 or not np.isfinite(rows).all():
            raise ValueError('HIGH_RATE_LAYOUT')
        if len(rows) and np.any(np.diff(rows[:, 0]) <= 0):
            raise ValueError('HIGH_RATE_CLOCK')
        self._highrate = (rows, nq, nv, nu, rows[:, 0].tolist())

    def _frame(self, row, index, fmt):
        obs = row['policy_obs'] if fmt == 'fixed_base' else row
        t = float(obs['timestamp_s'] if fmt == 'fixed_base' else row['time_s'])
        execution = row.get('execution', {})
        sup = row.get('supervision') or {}
        priv = row.get('privileged', {})
        frame = {'index': index, 'source_frame_index': self.source_indices[index], 'time_s': t, 'source_kind': self.kind,
                 'actual_tcp': pose(obs.get('actual_tcp', row.get('tcp'))),
                 'object_pose': pose(priv.get('object_pose', row.get('object_pose'))),
                 'desired_eef': pose(sup.get('desired')), 'accepted_eef': pose(sup.get('accepted')),
                 'committed_eef': pose(sup.get('committed')),
                 'actually_committed': sup.get('actually_committed'),
                 'actual_gripper': obs.get('actual_gripper', row.get('gripper_fraction')),
                 'reference_gripper': sup.get('committed_gripper'),
                 'teacher_stage': priv.get('teacher_stage', row.get('teacher_stage')),
                 'contacts': priv.get('contacts', row.get('contacts')),
                 'reference_pose_error_recorded': execution.get('reference_pose_error'),
                 'instruction': obs.get('instruction'), 'snapshot_id': row.get('snapshot_id'),
                 'qpos': row.get('qpos'), 'qvel': row.get('qvel'), 'ctrl': row.get('ctrl'),
                 'state_time_s': t if 'qpos' in row else None, 'cameras': {},
                 'offline_predictions': [], 'skeleton': None}
        if self._highrate is not None:
            rows, nq, nv, nu, times = self._highrate
            # Latest preceding physical record; no interpolation or future state.
            j = bisect.bisect_right(times, t + 1e-10) - 1
            if j >= 0:
                r = rows[j]
                frame.update(qpos=r[1:1+nq].tolist(), qvel=r[1+nq:1+nq+nv].tolist(),
                             ctrl=r[1+nq+nv:1+nq+nv+nu].tolist(), state_time_s=float(r[0]))
        for cam in CAMERAS:
            c = obs.get(cam)
            if c is None and fmt == 'native_body':
                name = f'images/{index:06d}-{cam}.png'
                # Require an exported camera index, never infer an image hash.
                c = self.spec.get('native_camera_index', {}).get(str(index), {}).get(cam)
                if c and c['path'] != name:
                    raise ValueError('NATIVE_CAMERA_INDEX_PATH')
            if c is None:
                frame['cameras'][cam] = None
                continue
            mapping = self.spec.get('image_path_map', {}).get(c['path'])
            image_root = self.root if mapping is None else self.image_roots[mapping['root']]
            path = local_file(image_root, c['path'] if mapping is None else mapping['path'])
            if c.get('frame_index', self.source_indices[index]) != self.source_indices[index]:
                raise ValueError('CAMERA_ORIGINAL_FRAME_INDEX_MISMATCH')
            ct = float(c['timestamp_s'])
            if not math.isfinite(ct) or abs(ct-t) > 1e-8:
                raise ValueError('CAMERA_STATE_CLOCK_MISMATCH')
            digest = c['sha256']
            if len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest):
                raise ValueError('CAMERA_SHA256_REQUIRED')
            self.assets[(index, cam)] = (path, digest, image_root)
            frame['cameras'][cam] = {'time_s': ct, 'sha256': digest, 'verified': False, 'original_path': c['path'], 'explicit_local_mapping': mapping is not None,
                                    'url': f'/api/asset/{self.id}/{index}/{cam}'}
        return finite(frame)

    def asset(self, index, camera):
        path, expected, authorized_root = self.assets[(index, camera)]
        # Recheck at read time, after any wait; never serve an arbitrary path.
        if not path.resolve().is_relative_to(authorized_root) or sha(path) != expected:
            raise ValueError('ORIGINAL_RGB_IDENTITY_CHANGED')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('RGB_CHANGED_DURING_READ')
        return raw

    def payload(self):
        return {'schema': 'EPISODE_REPLAY_VIEWER_V1', 'id': self.id, 'task': self.spec['task'],
                'source_kind': self.kind, 'fixture': self.fixture,
                'provenance': self.spec.get('provenance', {}), 'recorded_view': self.view,
                'original_audit': self.audit, 'original_audit_sha256': self.spec['audit_sha256'],
                'samples_sha256': self.samples_sha256, 'frames': self.frames, 'events': self.events,
                'limitations': ['Read-only saved facts; no simulation stepping, model, WBC or rescoring.',
                                'RGB SHA checked on each image request. Missing channels remain missing.',
                                'TCP/object markers are recorded facts; absent skeleton is not reconstructed.',
                                'Offline predictions are explicitly unexecuted overlays.']}


def add_offline_predictions(episode, overlay_spec):
    """Separate unexecuted predictions from actual saved state and task outcome."""
    p = local_file(episode.root, overlay_spec['path'])
    rows, _ = json_bytes(p, overlay_spec['sha256'])
    for row in rows:
        i = row['frame_index']
        if not isinstance(i, int) or i < 0 or i >= len(episode.frames):
            raise ValueError('OFFLINE_PREDICTION_FRAME')
        frame = episode.frames[i]
        if row['executed'] is not False or abs(float(row['observation_time_s'])-frame['time_s']) > 1e-8:
            raise ValueError('OFFLINE_PREDICTION_IS_UNEXECUTED_SAME_OBSERVATION_ONLY')
        points = row['xyz']
        if not isinstance(points, list) or any(not isinstance(v, list) or len(v) != 3 for v in points):
            raise ValueError('OFFLINE_PREDICTION_WORLD_XYZ_REQUIRED')
        if row['coordinate_frame'] != 'world':
            raise ValueError('OFFLINE_PREDICTION_FRAME_CONVERSION_MUST_BE_EXPLICIT')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for point in points for x in point):
            raise ValueError('OFFLINE_PREDICTION_NUMERIC_XYZ_REQUIRED')
        frame['offline_predictions'].append(finite({'label': row['label'], 'xyz': points,
            'executed': False, 'model_identity': row['model_identity'],
            'coordinate_frame': row['coordinate_frame'], 'original_overlay_sha256': overlay_spec['sha256']}))


def add_local_skeleton(episode, model_spec):
    """Load only an explicitly supplied local model; no asset distribution."""
    import mujoco
    from .kinematics import recorded_skeleton
    p = Path(model_spec['path']).resolve(strict=True)
    if sha(p) != model_spec['sha256']:
        raise ValueError('LOCAL_KINEMATIC_MODEL_SHA256_MISMATCH')
    # XML can reference more files. Require an explicit asset hash manifest and
    # read through MjModel's in-memory VFS so no undeclared filesystem is used.
    if p.suffix == '.mjb':
        model = mujoco.MjModel.from_binary_path(str(p))
    elif p.name.endswith('.xml.json'):
        doc = json.loads(p.read_text())
        xml = doc['xml']
        if hashlib.sha256(xml.encode()).hexdigest() != doc['sha256']:
            raise ValueError('EMBEDDED_SCENE_XML_SHA256_MISMATCH')
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(xml)
        assets = {}
        for a in model_spec.get('assets', []):
            name = a['name']
            if Path(name).is_absolute() or '..' in Path(name).parts or name in assets:
                raise ValueError('LOCAL_ASSET_VFS_NAME')
            ap = Path(a['path']).resolve(strict=True)
            if sha(ap) != a['sha256']:
                raise ValueError('LOCAL_ASSET_IDENTITY_CHANGED')
            assets[name] = ap.read_bytes()
        # Prohibit includes and plugins; this replay path accepts only a complete
        # saved scene and its explicitly listed mesh/texture bytes.
        if any(e.tag in ('include', 'plugin') for e in tree.iter()):
            raise ValueError('LOCAL_SCENE_MUST_BE_SELF_CONTAINED_NO_PLUGINS')
        compiler = tree.find('compiler')
        for key in ('meshdir', 'texturedir', 'assetdir'):
            if compiler is not None and compiler.get(key):
                raise ValueError('LOCAL_SCENE_ASSET_DIRECTORY_REQUIRES_EXPLICIT_VFS_EXPORT')
        for e in tree.iter():
            if e.get('file') is not None and e.get('file') not in assets:
                raise ValueError('LOCAL_SCENE_UNDECLARED_ASSET')
        model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    else:
        raise ValueError('LOCAL_MODEL_MUST_BE_MJB_OR_SAVED_SCENE_XML_JSON')
    for frame in episode.frames:
        if frame['qpos'] is not None:
            frame['skeleton'] = recorded_skeleton(model, frame, mujoco_module=mujoco)
