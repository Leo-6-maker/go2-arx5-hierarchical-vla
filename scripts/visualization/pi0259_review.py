"""Read sealed evidence; write a separate CPU review. Never import a producer runner."""
import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from eef_video_workflow import read, write, sha, text
from evidence_so3 import geodesic_angle, rot6d_rows_to_matrix
from plotting import plot, padded
from bundle_adapter import guard_render_imports

IMPORT_GUARD = guard_render_imports()

AUTH = '240dedfd3954dd4cbbb48645a2aad5b19738e2c4'
ALLOW = {'body_rgb', 'wrist_rgb', 'StateV1', 'language'}
ZERO = dict(model_loads=0, model_queries=0, training=0, scene_construction=0,
            physics=0, wbc_calls=0, hopper_jobs=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(root, relative):
    require(isinstance(relative, str) and relative and not re.match(r'^[A-Za-z]:', relative), 'invalid relative path')
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()) and p != root.resolve(), 'path escapes sealed packet')
    return p


def pose(a):
    a = np.asarray(a)
    require(a.shape[-2:] == (4, 4) and np.isfinite(a).all(), 'invalid SE3')
    require(np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0), 'invalid homogeneous row')
    r = a[..., :3, :3]
    require(np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-5, rtol=0)
            and np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0), 'invalid SO3')
    return a


class Packet:
    def __init__(self, manifest, digest):
        self.path = Path(manifest).resolve()
        require(sha(self.path) == digest, 'MANIFEST_HASH_MISMATCH')
        self.root = self.path.parent
        self.m = read(self.path)
        native_h0=self.m.get('schema')=='PI0259_H0_V1'
        require((native_h0 and self.m.get('producer_role')=='M' and self.m.get('review_state')=='PENDING_AGENT_V') or
                (self.m.get('sealed') is True and self.m.get('producer_role') == 'Agent M'), 'NOT_SEALED_AGENT_M_PACKET')
        require(self.m.get('protocol_commit',self.m.get('protocol_id')) == AUTH, 'AUTHORITY_MISMATCH')
        require(re.fullmatch('[0-9a-f]{40}', self.m.get('code_commit', '')), 'code commit missing')
        self.files = {}
        for item in self.m['entries' if native_h0 else 'files']:
            p = local(self.root, item['path'])
            require(item['path'] not in self.files, 'duplicate manifest path')
            require(p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'FILE_IDENTITY_MISMATCH: ' + item['path'])
            self.files[item['path']] = item
        require(self.files, 'empty packet')

    def file(self, name):
        require(name in self.files, 'unsealed file: ' + str(name))
        return local(self.root, name)

    def array(self, ref):
        # ponytail: explicit NPZ key bindings avoid a producer-specific import or executable mapping.
        with np.load(self.file(ref['path']), allow_pickle=False) as data:
            a = data[ref['key']].copy()
        return a

    def records(self):
        return read(self.file(self.m['review_records']))


def semantic_command(task, instruction, colors):
    words = re.findall(r'[a-z]+', instruction.lower())
    if task == 'M1':
        hits = [i for i, c in enumerate(colors) if c in words]
        require(len(hits) == 1 and set(words) & {'pose', 'axes', 'orientation', 'frame'}, 'M1 command ambiguous')
        return hits[0]
    if task == 'M2':
        require('orange' in words and set(words) & {'block', 'obstacle'}, 'M2 semantics absent')
        require(('left' in words) != ('right' in words), 'M2 command ambiguous')
        return 0 if 'left' in words else 1
    require(task == 'M3' and 'above' in words and 'black' in words and 'white' in words, 'M3 semantics absent')
    require(('clockwise' in words) != ('counterclockwise' in words), 'M3 command ambiguous')
    return 0 if 'counterclockwise' in words else 1


def project(points, view, projection, convention):
    require(convention in ('row_vector_opengl', 'column_vector_opengl'), 'unsupported camera convention')
    points = np.c_[points, np.ones(len(points))]
    if convention == 'column_vector_opengl':
        view, projection = view.T, projection.T
    camera = points @ view
    clip = camera @ projection
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
    ndc = np.full((len(points), 3), np.nan)
    ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
    pixel = np.c_[(ndc[:, 0] + 1) * 48, (1 - ndc[:, 1]) * 48]
    return pixel, valid & (np.abs(ndc).max(axis=1) <= 1)


def check_record(packet, row):
    require(row['task'] in ('M1', 'M2', 'M3') and row['command'] in (0, 1), 'unknown task/command')
    require(semantic_command(row['task'], row['instruction'], row['colors']) == row['command'], 'SWAPPED_COMMAND')
    target = ('target' + str(row['command'])) if row['task']=='M1' else 'goal' if row['task']=='M2' else ('white_B' if row['command']==0 else 'black_A')
    require(row['target_id'] == target == row['command_targets'][str(row['command'])], 'CHANGED_TARGET_ID')
    require(row['frame'] == 'world' and row['state_frame'] == 'native', 'MIXED_NATIVE_WORLD_FRAME')
    require(row['units'] == {'position': 'm', 'rotation': 'rad', 'time': 's'}, 'unknown units')
    require(row['state_origin'] == 'KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT', 'unknown state origin')
    require(row['time_origin'] == 'synthetic_reference_time', 'unknown clock')
    require(row['rgb_order'] in ('RGB', 'BGR'), 'RGB/BGR unspecified')
    a = {k: packet.array(v) for k, v in row['arrays'].items()}
    t = a['time']; n = len(t)
    require(t.shape == (n,) and n > 0 and np.isfinite(t).all() and np.all(np.diff(t) > 0), 'invalid time')
    fk, reference, anchor = pose(a['fk_world']), pose(a['reference_world']), pose(a['T_world_native'])
    require(fk.shape == reference.shape == (n, 4, 4) and anchor.shape == (4, 4), 'pose/time shape mismatch')
    state = a['StateV1']
    require(state.shape == (n, 10) and np.isfinite(state).all(), 'StateV1 shape/nonfinite')
    native = np.linalg.inv(anchor) @ fk
    require(np.allclose(state[:, :3], native[:, :3, 3], atol=1e-5, rtol=0), 'STATE_FK_POSITION_MISMATCH')
    require(np.max(geodesic_angle(rot6d_rows_to_matrix(state[:, 3:9]), native[:, :3, :3])) <= 1e-4, 'STATE_FK_ROTATION_MISMATCH')
    require(np.all(state[:, 9] == .5), 'gripper not neutral')
    # Independent numeric FK: parse only sealed URDF and saved joints; no assets or scene constructor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pi0258_geometry import full_fk
    from go2arx5_vla.wbc.native.kinematics import homogeneous
    urdf = ET.parse(packet.file(row['urdf'])).getroot()
    movable = {j.attrib['name'] for j in urdf.findall('joint') if j.attrib['type'] != 'fixed'}
    require(movable <= set(row['joint_names']), 'missing movable joint DOFs')
    require(a['joints'].shape == (n, len(row['joint_names'])), 'DOF shape mismatch')
    require(a['root_world'].shape == (n, 4, 4), 'root/time shape mismatch')
    roots = pose(a['root_world']); recomputed = []; body = []; wrist = []
    for q, root in zip(a['joints'], roots):
        poses = full_fk(urdf, dict(zip(row['joint_names'], q)), root)
        recomputed.append(poses[row['tcp_link']] @ homogeneous(np.eye(3), np.array(row['tcp_offset_m'])))
        body.append(poses[row['camera_links']['body']]); wrist.append(poses[row['camera_links']['wrist']])
    require(np.allclose(recomputed, fk, atol=1e-5, rtol=0), 'SAVED_FK_URDF_MISMATCH')
    require(np.allclose(pose(a['render_tcp_world']), fk, atol=1e-5, rtol=0), 'RENDER_STATE_POSE_MISMATCH')
    if abs(t[0]) < 1e-9:
        require(np.allclose(reference[0], fk[0], atol=1e-5, rtol=0), 'INITIAL_REFERENCE_NOT_FK')
    cues = row['cues']; ids = [c['id'] for c in cues]
    require(len(ids) == len(set(ids)), 'duplicate cue IDs')
    cue_counts = {}; projections = {}; rgb_means = {}; centroid_errors = {}
    for stream, link_poses in [('body', body), ('wrist', wrist)]:
        rgb, seg, depth = [a[stream + '_' + k] for k in ('rgb', 'segmentation', 'depth')]
        require(rgb.shape == (n, 96, 96, 3) and rgb.dtype == np.uint8, 'not source 96x96 uint8 RGB')
        require(seg.shape == depth.shape == (n, 96, 96) and np.issubdtype(seg.dtype, np.integer), 'mask/depth dimensions or dtype')
        require(a[stream + '_time'].shape == t.shape and np.array_equal(a[stream + '_time'], t), 'IMAGE_STATE_TIME_MISMATCH')
        camera = pose(a[stream + '_world'])
        require(np.allclose(camera, np.asarray(link_poses) @ pose(row['camera_attachments'][stream]), atol=1e-5, rtol=0), 'CAMERA_FK_ATTACHMENT_MISMATCH')
        view, projection = a[stream + '_view'], a[stream + '_projection']
        require(view.shape == projection.shape == (n, 4, 4) and np.isfinite(projection).all(), 'camera matrix shape')
        expected_view = np.linalg.inv(camera)
        expected_view = expected_view.swapaxes(-1, -2) if row['camera_convention'] == 'row_vector_opengl' else expected_view
        require(np.allclose(view, expected_view, atol=1e-5, rtol=0), 'CAMERA_VIEW_WORLD_MISMATCH')
        require(row['depth_convention'] in ('negative_camera_z_m', 'positive_camera_z_m'), 'depth convention missing')
        cue_counts[stream] = {}; projections[stream] = {}; rgb_means[stream] = {}; centroid_errors[stream] = {}
        for cue in cues:
            mask = seg == cue['id']; counts = mask.sum(axis=(1, 2)); cue_counts[stream][cue['name']] = counts.tolist()
            require(np.isfinite(depth[mask]).all(), 'cue has invalid depth')
            require(np.all(depth[mask] < 0) if row['depth_convention'].startswith('negative') else np.all(depth[mask] > 0), 'cue depth sign mismatch')
            colors = rgb[mask][:, ::-1] if row['rgb_order'] == 'BGR' else rgb[mask]
            rgb_means[stream][cue['name']] = colors.mean(axis=0).tolist() if len(colors) else None
            projected = [project(np.asarray([cue['world_m']]), v, p, row['camera_convention'])[0][0].tolist() for v, p in zip(view, projection)]
            projected = [p if np.isfinite(p).all() else None for p in projected]
            projections[stream][cue['name']] = projected
            centroid_errors[stream][cue['name']] = [float(np.linalg.norm(np.argwhere(m)[:,::-1].mean(axis=0)+.5-p)) if m.any() and p is not None else None for m,p in zip(mask,projected)]
    require(set(row['required_cues']) <= {c['name'] for c in cues}, 'required cue absent from ID map')
    visible = np.ones(n, dtype=bool)
    for cue in cues:
        if cue['name'] not in row['required_cues']:
            continue
        threshold = 16 if cue['role'] == 'obstacle' else 2 if cue['role'] == 'axis' else 4
        counts = np.maximum(cue_counts['body'][cue['name']], cue_counts['wrist'][cue['name']])
        visible &= counts >= threshold
    if row['task']=='M3':
        sectors = [c for c in cues if c['role']=='contour_sector']
        require({c['sector'] for c in sectors}==set(range(12)), 'M3 ordered sector mapping missing')
        sector_visible=np.array([np.maximum(cue_counts['body'][c['name']],cue_counts['wrist'][c['name']])>0 for c in sectors])
        if abs(t[0])<1e-9: require(int(sector_visible[:,0].sum())>=8, 'M3_START_CONTOUR_INVISIBLE')
    active = a['active'].astype(bool)
    require(active.shape == (n,) and active.any(), 'active coverage missing')
    err = np.linalg.norm(fk[:, :3, 3] - reference[:, :3, 3], axis=1)
    rot = geodesic_angle(fk[:, :3, :3], reference[:, :3, :3])
    duration = {'M1': 6., 'M2': 10., 'M3': 16.}[row['task']]
    quarters = np.minimum((t / duration * 4).astype(int), 3)
    require(set(quarters[active]) == {0, 1, 2, 3} if row.get('complete_sequence') else True, 'DROPPED_MOTION_QUARTER')
    diagnostics = {}
    if 'joints' in a and n>1:
        diagnostics['peak_joint_velocity_rad_s']=np.max(np.abs(np.diff(a['joints'],axis=0)/np.diff(t)[:,None]),axis=0).tolist()
    for field in ('solver_position_residual_m','solver_rotation_residual_rad','collision_flags','velocity_flags','raw_gate_reject','codec_gate_reject'):
        diagnostics[field] = a[field].tolist() if field in a else 'MISSING'
    return dict(source_id=row['source_id'], arrays=a, cue_counts=cue_counts, projected_centers_px=projections,
                projected_centroid_distance_px=centroid_errors,diagnostics=diagnostics,
                cue_rgb_means=rgb_means, visibility_fraction=float(visible[active].mean()),
                position_max_m=float(err.max()), rotation_max_rad=float(rot.max()),
                qualified_pose=(err <= .002) & (rot <= .01), quarters=quarters,
                visible=visible, active=active,
                limits='Raster counts are necessary only. Projection centers do not establish visibility; natural RGB discernibility and source camera calibration need explicit review.')


def check_cohort(rows, results, packet):
    geometry_splits = {}; pair_rows = {}; counts = {}; issues = []
    for row in rows:
        # Complete geometry identity deliberately excludes color, task, language and family name.
        key = json.dumps([row['geometry_tuple'], row['fixed_root_dofs_sha256'], row['camera_sha256'], row['asset_sha256']], sort_keys=True)
        if key in geometry_splits and geometry_splits[key] != row['split']:
            issues.append('GEOMETRY_SPLIT_LEAK:' + row['source_id'])
        geometry_splits[key] = row['split']
        if row['split'].startswith('TEST'):
            issues.append('UNAUTHORIZED_TEST_ARRAYS:' + row['source_id'])
        require(set(row['collator_fields']) == ALLOW, 'TARGET_METADATA_OR_DIAGNOSTICS_IN_COLLATOR')
        # Actual saved collator values must have the same RGB/State bytes; no declared PASS shortcuts.
        r = results.get(row['source_id'])
        if not r:
            continue
        for field in ALLOW - {'language'}:
            require(np.array_equal(packet.array(row['collator_arrays'][field]), r['arrays'][field]), 'COLLATOR_VALUE_MISMATCH')
        require(row['collator_language'] == row['instruction'], 'COLLATOR_LANGUAGE_MISMATCH')
        pair_rows.setdefault((key, row['task'], row['split']), []).append(row)
        a = r['arrays']; starts = a['window_start_index'].astype(int)
        require(starts.ndim == 1 and len(starts) > 0 and np.all(starts >= 0) and np.all(starts < len(a['time'])), 'bad window index')
        raw = a['raw_gate_reject']; codec = a['codec_gate_reject']
        require(raw.shape == codec.shape == starts.shape, 'gate/window mismatch')
        good = a['window_geometry_valid'].astype(bool) & ~raw.astype(bool) & ~codec.astype(bool) & r['visible'][starts]
        usable = bool(row['complete_sequence'] and good.mean() >= .95 and set(r['quarters'][starts[good]]) == {0, 1, 2, 3})
        # These window geometry/gate fields are reported separately, not independently replayed here.
        counts.setdefault(row['task'], {}).setdefault(row['split'], dict(attempts=0, array_qualified=0))
        counts[row['task']][row['split']]['attempts'] += 1
        counts[row['task']][row['split']]['array_qualified'] += int(usable)
    complete_pairs = {}
    for (key, task, split), pair in pair_rows.items():
        if len(pair) != 2 or {r['command'] for r in pair} != {0, 1}:
            issues.append('MISSING_OR_DUPLICATE_COMMAND_PAIR:' + pair[0]['source_id']); continue
        left, right = [results[r['source_id']]['arrays'] for r in pair]
        li, ri = [np.asarray(r['paired_prefix_indices'], dtype=int) for r in pair]
        require(len(li) == len(ri) > 0, 'EMPTY_PAIRED_PREFIX')
        for field in ('body_rgb', 'wrist_rgb', 'StateV1', 'time'):
            require(np.array_equal(left[field][li], right[field][ri]), 'NON_PAIRED_INPUT:' + field)
        complete_pairs[task + '/' + split] = complete_pairs.get(task + '/' + split, 0) + 1
    return dict(counts=counts, complete_input_pairs=complete_pairs, findings=issues,
                thresholds=dict(development_usable=10,development_attempts=12,train_usable=64,train_attempts=80,validation_usable=16,validation_attempts=20,train_pairs=24,validation_pairs=6,minimum_tasks=2),
                limitation='H1 full PASS additionally requires original gate/collision replay, fixed registry/split algorithm, TRAIN-only statistics and actual collator source audit; these are never inferred from fields.')


def prediction_card(row, result, path, digest):
    a=result['arrays'];canvas=np.full((720,1280,3),20,np.uint8)
    text(canvas,'OFFLINE MODEL PREDICTION - NO WBC EXECUTION',20,30,.7)
    text(canvas,f"{row['model']} step {row['step']} | {row['task']} | {row['source_id']}",20,62,.5)
    text(canvas,row['instruction'],20,88,.45)
    text(canvas,'checkpoint '+row['checkpoint_sha256'],20,115,.42)
    text(canvas,'noise '+row['noise_sha256'],20,140,.42)
    time=a['time'];p=a['prediction'][:,:3,3];r=a['reference'][:,:3,3]
    for axis in range(3):
        plot(canvas,(20+axis*420,170,410,290),[(time,p[:,axis],(100,200,250)),(time,r[:,axis],(120,230,130))],
             (0,1.95),padded(np.r_[p[:,axis],r[:,axis]]),f'native {"XYZ"[axis]} (m), prediction/reference','s','m')
    text(canvas,f"XYZ RMSE {result['xyz_rmse_m']*1000:.3f} mm | mean SO3 {result['so3_mean_rad']:.5f} rad",20,510,.55)
    text(canvas,'Saved decoded arrays only; source CPU scorer and semantics require separate review.',20,550,.46)
    text(canvas,'Individual panel scales explicitly labelled; not a common-scale causal comparison.',20,580,.46)
    text(canvas,'manifest SHA256 '+digest,20,685,.4)
    require(cv2.imwrite(str(path),canvas),'card write failed')


def card(row, result, frame, path, manifest_hash):
    a = result['arrays']; canvas = np.full((720, 1280, 3), 20, np.uint8)
    text(canvas, 'KINEMATIC REFERENCE - NO WBC EXECUTION', 20, 25, .65)
    text(canvas, row['source_id'] + ' | ' + row['task'] + ' | command ' + str(row['command']), 20, 52)
    text(canvas, row['instruction'], 20, 78, .45)
    text(canvas, f"t={a['time'][frame]:.3f}s synthetic | 96x96 source; nearest display 3x", 20, 104, .48)
    for j, stream in enumerate(('body', 'wrist')):
        x = 20 + j * 320; rgb = a[stream + '_rgb'][frame]
        if row['rgb_order'] == 'RGB': rgb = rgb[..., ::-1]
        canvas[142:430, x:x+288] = cv2.resize(rgb, (288, 288), interpolation=cv2.INTER_NEAREST)
        text(canvas, stream + ' POLICY RGB', x, 133, .5)
        seg = a[stream + '_segmentation'][frame]
        # Diagnostic palette is derived from integer IDs and never written over RGB.
        mask = np.stack([(seg * k) % 251 for k in (31, 71, 131)], axis=-1).astype(np.uint8)
        canvas[472:664, x:x+192] = cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST)
        text(canvas, 'DIAGNOSTIC mask', x, 460, .43)
        d = np.abs(a[stream + '_depth'][frame]); finite = np.isfinite(d)
        normalized = np.zeros(d.shape, np.uint8)
        if finite.any(): normalized[finite] = np.clip(d[finite] / max(float(d[finite].max()), 1e-6) * 255, 0, 255).astype(np.uint8)
        canvas[472:568, x+196:x+292] = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        text(canvas, 'depth (m)', x+196, 590, .32)
    err = np.linalg.norm(a['fk_world'][:, :3, 3]-a['reference_world'][:, :3, 3], axis=1)*1000
    t = a['time']; xlim=(float(t[0]), max(float(t[-1]), float(t[0])+.05))
    plot(canvas,(665,130,595,260),[(t,err,(100,200,250))],xlim,padded(err,1),'FK vs requested, same world/timestamp','s','mm',cursor=float(t[frame]),threshold=2)
    text(canvas, 'Raster counts body/wrist (not projection-box claims):', 680, 430, .43)
    for i, cue in enumerate(row['required_cues'][:9]):
        b=result['cue_counts']['body'][cue][frame];w=result['cue_counts']['wrist'][cue][frame]
        text(canvas, f'{cue}: {b}/{w} px', 680, 454+20*i, .4)
    text(canvas, 'Full counts and failures: receipt.json. Natural RGB review still required.', 20, 689, .4)
    text(canvas, 'manifest SHA256 ' + manifest_hash, 20, 712, .38)
    require(cv2.imwrite(str(path), canvas), 'card write failed')


def review(manifest, digest, out, stage):
    packet = Packet(manifest, digest)
    out = Path(out).resolve()
    require(not out.is_relative_to(packet.root) and not packet.root.is_relative_to(out), 'OUTPUT_OVERLAPS_SEALED_PACKET')
    require(not out.exists(), 'new companion output directory required')
    require(packet.m['stage'] == stage, 'stage mismatch')
    out.mkdir(parents=True)
    receipt = dict(schema='PI0259_AGENT_V_REVIEW_V1',stage=stage,producer_manifest_sha256=digest,
                   reviewer_code_sha256=sha(Path(__file__)),new_calls=ZERO,findings=[],results=[],cards=[],
                   selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    try:
        rows = packet.records(); results = {}
        require(rows and len({r['source_id'] for r in rows}) == len(rows), 'empty/duplicate source IDs')
        require(set(packet.m['source_ids']) == {r['source_id'] for r in rows}, 'incomplete outcome index')
        for row in rows:
            try:
                if stage == 'H2':
                    result = prediction_record(packet, row)
                else:
                    result = check_record(packet, row)
                results[row['source_id']] = result
                receipt['results'].append({k:v for k,v in result.items() if k not in ('arrays','qualified_pose','quarters','visible','active')})
            except (KeyError, ValueError, TypeError, IndexError, OSError) as error:
                receipt['findings'].append(dict(source_id=row['source_id'],error=str(error),originals=row.get('arrays',{})))
        if stage == 'H1': receipt['cohort'] = check_cohort(rows, results, packet)
        if stage == 'H2':
            groups={}
            for row in sorted(rows,key=lambda r:hashlib.sha256(r['source_id'].encode()).hexdigest()):
                result=results.get(row['source_id'])
                if result is None:continue
                name=f'prediction-{len(receipt["cards"]):04d}.png'
                prediction_card(row,result,out/name,digest)
                receipt['cards'].append(dict(path=name,source_id=row['source_id'],sha256=sha(out/name)))
                groups.setdefault((row['model'],row['step'],row['task']),[]).append(result['xyz_rmse_m'])
            receipt['per_task']=[dict(model=m,step=s,task=t,count=len(v),median_xyz_rmse_m=float(np.median(v))) for (m,s,t),v in groups.items()]
            receipt['worst_task']=[dict(model=m,step=s,worst_task_median_xyz_rmse_m=max(x['median_xyz_rmse_m'] for x in receipt['per_task'] if x['model']==m and x['step']==s)) for m,s in sorted({(r['model'],r['step']) for r in rows}) if any(x['model']==m and x['step']==s for x in receipt['per_task'])]
        # Metadata-first representatives, independently of model metrics: every task/command/quarter.
        if stage != 'H2':
            ordered = sorted(rows,key=lambda r:hashlib.sha256(r['source_id'].encode()).hexdigest())
            chosen = set()
            for row in ordered:
                result = results.get(row['source_id'])
                if result is None: continue
                for quarter in sorted(set(result['quarters'].tolist())):
                    key=(row['task'],row['command'],int(quarter),row.get('outcome','MISSING_OUTCOME'))
                    if key in chosen: continue
                    chosen.add(key);frame=int(np.flatnonzero(result['quarters']==quarter)[0])
                    name=f'card-{len(receipt["cards"]):04d}.png'
                    card(row,result,frame,out/name,digest)
                    receipt['cards'].append(dict(path=name,source_id=row['source_id'],frame=frame,sha256=sha(out/name)))
        receipt['status'] = 'CORRECTNESS_BLOCKED' if receipt['findings'] or receipt.get('cohort',{}).get('findings') else stage+'_NUMERIC_REVIEW_PARTIAL'
        receipt['closure_required'] = ['Natural RGB/source calibration review for discernibility and cue/color identities.',
          'Independent source-bound solver/collision/gate and full registry/processor audit as applicable; attach exact hash-bound receipt, never producer PASS.',
          'H2 semantic/paraphrase denominators and CPU scorer bindings; this adapter is not a deployment selector.']
    except (KeyError, ValueError, TypeError, IndexError, OSError) as error:
        receipt['status']='CORRECTNESS_BLOCKED';receipt['findings'].append(dict(error=str(error)))
    receipt['original_files'] = list(packet.files.values())
    write(out/'receipt.json',receipt)
    parts=['<!doctype html><meta charset="utf-8"><title>PI0259 independent review</title>', '<h1>'+receipt['status']+'</h1>',
           '<p>KINEMATIC REFERENCE — NO WBC EXECUTION / OFFLINE MODEL PREDICTION. Historical FAIL/NONE unchanged.</p>',
           '<p>Manifest SHA256: '+digest+'</p><a href="receipt.json">Full results, failures, missing checks and original hashes</a>']
    for c in receipt['cards']:
        parts.append('<p>'+html.escape(c['source_id'])+'</p><img width="1280" src="'+c['path']+'">')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    return receipt


def prediction_record(packet, row):
    require(row['evidence_class']=='OFFLINE_MODEL_PREDICTION', 'H2 class mismatch')
    require(row['step'] in (0,6000) and row['model'] in ('MT-Base','MT-G'), 'unapproved model endpoint')
    for key in ('checkpoint_sha256','processor_sha256','initializer_revision','noise_sha256','cpu_receipt'):
        require(row.get(key), 'missing identity '+key)
    for key in ('checkpoint_sha256','processor_sha256','noise_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',row[key]), 'invalid identity '+key)
    require(re.fullmatch('[0-9a-f]{40}',row['initializer_revision']), 'initializer revision missing')
    require(row['first_valid_or_reuse'] in ('FIRST_VALID','EXACT_REUSE'), 'prediction outcome status absent')
    require(np.isfinite(row['inference_latency_s']) and row['inference_latency_s']>=0, 'latency absent')
    packet.file(row['cpu_receipt'])
    a={k:packet.array(v) for k,v in row['arrays'].items()}
    require(row['frame']=='native' and a['prediction'].shape==a['reference'].shape==(40,4,4), 'H2 frame/horizon')
    prediction,reference=pose(a['prediction']),pose(a['reference'])
    require(np.array_equal(a['time'],np.arange(40)*.05), 'H2 clock mismatch')
    require(hashlib.sha256(a['noise'].tobytes()).hexdigest()==row['noise_sha256'], 'H2 noise mismatch')
    err=np.linalg.norm(prediction[:,:3,3]-reference[:,:3,3],axis=1)
    rot=geodesic_angle(prediction[:,:3,:3],reference[:,:3,:3])
    return dict(source_id=row['source_id'],model=row['model'],step=row['step'],task=row['task'],noise_sha256=row['noise_sha256'],
                xyz_rmse_m=float(np.sqrt(np.mean(err**2))),so3_mean_rad=float(rot.mean()),arrays=a,
                metrics_source='Independent CPU from sealed decoded SE3 arrays; no model or decoder invoked.',
                limitation='Does not independently prove raw-output decoder/gate or task semantics. CPU receipt requires separate source review.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='plan',choices=['plan','H0','H1','H2','run-episode'])
    p.add_argument('--manifest',type=Path);p.add_argument('--sha256');p.add_argument('--out',type=Path)
    p.add_argument('--urdf',type=Path,help='private local URDF; must match producer source binding, never copied into output')
    a=p.parse_args()
    if a.stage=='run-episode':p.error('GUARD_STUB_ONLY: physical runner NOT_CONNECTED')
    if a.stage=='plan':
        print(json.dumps(dict(status='WAITING_FOR_SEALED_HANDOFF',authority=AUTH,new_calls=ZERO,entry=['H0','H1','H2'])));return
    if not all([a.manifest,a.sha256,a.out]):p.error('sealed --manifest, externally supplied --sha256 and separate --out are required')
    try:
        if read(a.manifest).get('schema')=='PI0259_H0_V1':
            require(a.stage=='H0' and a.urdf is not None,'H0 native packet requires hash-bound --urdf')
            from pi0259_h0 import review_h0
            r=review_h0(a.manifest,a.sha256,a.out,a.urdf)
        else:r=review(a.manifest,a.sha256,a.out,a.stage)
        print(json.dumps(dict(status=r['status'],receipt=str(a.out/'receipt.json'))))
    except (ValueError,KeyError,OSError) as error:p.error(str(error))


if __name__=='__main__':main()
