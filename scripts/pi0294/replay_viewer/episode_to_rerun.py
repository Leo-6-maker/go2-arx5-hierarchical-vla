"""Thin read-only PI0294 adapter to the official Rerun viewer/file format.

The SDK owns playback, timeline, cameras, 3D and plots. No custom player, physics,
policy, WBC or rescoring is implemented here. Private RRDs contain actual RGB.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import uuid

from .records import Episode, add_offline_predictions, add_local_skeleton, sha

SDK_VERSION = '0.40.0'
APPLICATION_ID = 'go2-arx5-recorded-episode-replay-v1'
TIME = 'recorded_time'
FRAME = 'source_frame'


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _xyz(pose):
    return [row[3] for row in pose[:3]]


def load_episodes(config, *, allow_local_kinematics=False):
    episodes = []
    for spec in json.loads(Path(config).read_text())['episodes']:
        e = Episode(spec)
        if any(x.id == e.id for x in episodes):
            raise ValueError('DUPLICATE_EPISODE_ID')
        if spec.get('offline_predictions'):
            add_offline_predictions(e, spec['offline_predictions'])
        if spec.get('kinematic_model'):
            if not allow_local_kinematics:
                raise PermissionError('LOCAL_MODEL_LOADING_REQUIRES_EXPLICIT_FLAG')
            add_local_skeleton(e, spec['kinematic_model'])
        episodes.append(e)
    if not episodes:
        raise ValueError('NO_SAVED_EPISODES')
    return episodes


def blueprint(episode):
    import rerun.blueprint as rrb
    # Configure the official orbital camera from saved physical/command poses.
    # This changes only initial viewing, never coordinates, state or playback.
    points = [_xyz(f[k]) for f in episode.frames for k in
              ('actual_tcp', 'object_pose', 'desired_eef', 'accepted_eef', 'committed_eef') if f[k] is not None]
    if points:
        low = [min(p[k] for p in points) for k in range(3)]
        high = [max(p[k] for p in points) for k in range(3)]
        center = [(a+b)/2 for a,b in zip(low,high)]
        span = max(.15, *(b-a for a,b in zip(low,high)))
    else:
        center, span = [0.,0.,0.], .5
    eye = [center[k] + span*d for k,d in enumerate((2., -2.5, 1.8))]
    controls = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=eye,
                                look_target=center, eye_up=[0.,0.,1.])
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin='/episode/rgb/body_rgb', name='Recorded BODY RGB'),
                rrb.Spatial2DView(origin='/episode/rgb/wrist_rgb', name='Recorded WRIST RGB')),
            rrb.Horizontal(
                rrb.Spatial3DView(origin='/episode/world', name='Recorded poses / sparse points; no simulation', eye_controls=controls),
                rrb.TimeSeriesView(origin='/episode/signals', name='Saved gripper and source state')),
            rrb.Horizontal(
                rrb.TextDocumentView(origin='/episode/original_audit', name='Original audit / provenance / limits'),
                rrb.TextDocumentView(origin='/episode/current_frame', name='Saved current frame (full precision)'),
                rrb.TextLogView(origin='/episode/events', name='Original recorded stage / S1 events')),
            row_shares=[2, 3, 2]),
        rrb.TimePanel(timeline=TIME, expanded=True, play_state='paused'),
        collapse_panels=True)


def _set_frame_time(rec, frame):
    rec.set_time(TIME, duration=frame['time_s'])
    rec.set_time(FRAME, sequence=frame['source_frame_index'])


def _log_events(rec, rr, episode):
    # Never invent a time for an S1 record which only saved a source sequence.
    # Both original JSON and exact available timeline index are retained.
    times = {f['source_frame_index']: f['time_s'] for f in episode.frames}
    for source in episode.events:
        rows = source['original_records']
        if not isinstance(rows, list):
            continue
        for row in rows:
            rec.disable_timeline(TIME)
            rec.disable_timeline(FRAME)
            t = row.get('time_s')
            sequence = row.get('sequence')
            if t is not None:
                rec.set_time(TIME, duration=float(t))
            if isinstance(sequence, int):
                rec.set_time(FRAME, sequence=sequence)
                if t is None and sequence in times:
                    rec.set_time(TIME, duration=times[sequence])
            rec.log('episode/events/'+source['file'].replace('.json', ''),
                    rr.TextLog(_json(row), level='WARN' if row.get('decision') == 'REJECT' else 'INFO'))
    rec.disable_timeline(TIME)
    rec.disable_timeline(FRAME)


def export_episode(episode, output_directory):
    """Export one private recording; never mutate input content or permissions."""
    previous = os.umask(0o077)
    try:
        return _export_episode(episode, output_directory)
    finally:
        os.umask(previous)


def _export_episode(episode, output_directory):
    import rerun as rr
    if version('rerun-sdk') != SDK_VERSION:
        raise RuntimeError('PINNED_RERUN_SDK_REQUIRED:'+SDK_VERSION)
    output = Path(output_directory)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    rrd = output/(episode.id+'.rrd')
    rbl = output/(episode.id+'.rbl')
    rid = str(uuid.uuid4())
    rec = rr.RecordingStream(APPLICATION_ID, recording_id=rid)
    rec.save(rrd)
    bp = blueprint(episode)
    bp.save(APPLICATION_ID, rbl)
    rec.send_blueprint(bp)
    payload = episode.payload()
    metadata = {k: v for k, v in payload.items() if k not in ('frames', 'events')}
    metadata['original_event_records'] = episode.events
    metadata['viewer'] = 'Official Rerun; no custom playback, timeline or renderer'
    metadata['sparse_interpolation'] = 'None: source points only; viewer holds the latest saved sample between times'
    metadata['joint_state_availability'] = {'qpos_frames': sum(f['qpos'] is not None for f in episode.frames), 'qvel_frames': sum(f['qvel'] is not None for f in episode.frames), 'missing_qpos_is_not_filled_with_arm_joint_subset': True}
    metadata['robot_geometry'] = 'No invented skeleton/mesh; optional skeleton only from supplied legal model and recorded qpos'
    subset = episode.view.get('kind') == 'recorded_keyframe_subset'
    scope = f"KEYFRAME SUBSET {len(episode.frames)}/{episode.view['source_frame_count']} — NOT CONTINUOUS REPLAY" if subset else 'Saved observation frames only'
    header = f'# {episode.kind} · {episode.id}\n\n**{scope}**\n\nOriginal audit below; no rescoring. Missing robot assets/joint state are not invented.\n\n'
    rec.log('episode/original_audit', rr.TextDocument(header+'```json\n'+_json(metadata)+'\n```', media_type='text/markdown'), static=True)
    rec.log('episode/world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for name in ('actual_gripper', 'reference_gripper', 'reference_pose_error_recorded'):
        rec.log('episode/signals/'+name, rr.SeriesPoints(), static=True)
    camera_count = image_bytes = 0
    histories = {k: [] for k in ('actual_tcp', 'object_pose', 'desired_eef', 'committed_eef')}
    colors = {'actual_tcp': [65,210,180], 'object_pose': [100,145,240],
              'desired_eef': [240,180,70], 'accepted_eef': [200,155,65], 'committed_eef': [245,200,100]}
    for frame in episode.frames:
        _set_frame_time(rec, frame)
        rec.log('episode/current_frame', rr.TextDocument(_json(frame)))
        for cam in ('body_rgb', 'wrist_rgb'):
            record = frame['cameras'][cam]
            if record is None:
                rec.log('episode/rgb/'+cam, rr.Clear(recursive=True))
                continue
            image = episode.asset(frame['index'], cam)
            rec.log('episode/rgb/'+cam, rr.EncodedImage(contents=image, media_type='image/png'))
            camera_count += 1
            image_bytes += len(image)
        for name, color in colors.items():
            pose = frame[name]
            if pose is None:
                rec.log('episode/world/'+name, rr.Clear(recursive=True))
                continue
            point = _xyz(pose)
            rec.log('episode/world/'+name, rr.Points3D([point], colors=color, radii=.004, labels=[name]))
            if name in histories:
                histories[name].append(point)
                # A point history avoids interpolating unobserved sparse motion.
                rec.log('episode/world/saved_points/'+name, rr.Points3D(histories[name], colors=color, radii=.0015))
        for name in ('actual_gripper', 'reference_gripper', 'reference_pose_error_recorded'):
            value = frame[name]
            if value is not None:
                rec.log('episode/signals/'+name, rr.Scalars(value))
            else:
                rec.log('episode/signals/'+name, rr.Clear(recursive=True))
        if frame.get('skeleton'):
            sk = frame['skeleton']
            rec.log('episode/world/recorded_fk_skeleton', rr.LineStrips3D([[sk['points'][a], sk['points'][b]] for a,b in sk['edges']], colors=[190,200,220]))
        rec.log('episode/world/offline_prediction_NOT_EXECUTED', rr.Clear(recursive=True))
        for i, pred in enumerate(frame['offline_predictions']):
            if pred['executed'] is not False:
                raise ValueError('UNEXECUTED_OVERLAY_REQUIRED')
            rec.log(f'episode/world/offline_prediction_NOT_EXECUTED/{i}',
                    rr.Points3D(pred['xyz'], colors=[195,115,220], radii=.003, labels=[pred['label']+' NOT EXECUTED']))
        rec.log('episode/events/saved_stage', rr.TextLog('Saved stage: '+str(frame['teacher_stage'])))
    _log_events(rec, rr, episode)
    rec.flush()
    rec.disconnect()
    receipt = {'schema':'EPISODE_TO_RERUN_EXPORT_V1', 'sdk_version':SDK_VERSION,
        'application_id':APPLICATION_ID, 'recording_id':rid, 'episode_id':episode.id,
        'source_kind':episode.kind, 'fixture':episode.fixture, 'recorded_view':episode.view,
        'exported_observations':len(episode.frames), 'RGB_records':camera_count, 'RGB_encoded_bytes':image_bytes,
        'audit_sha256':episode.spec['audit_sha256'], 'selected_samples_sha256':episode.samples_sha256,
        'recording_sha256':sha(rrd), 'blueprint_sha256':sha(rbl),
        'new_integrations':0, 'new_model_queries':0, 'new_WBC_calls':0, 'task_rescoring':False,
        'source_files_modified':False, 'semantic_recording_readback':'NOT_YET_RUN',
        'limits':['Original audit is historical evidence, not a new score',
                  'Sparse samples are not continuous reconstructed motion',
                  'Rerun may hold the last saved sample between observations; no new state is fabricated',
                  'RRD contains private actual RGB and must remain local; no public upload']}
    (output/'EXPORT_RECEIPT.private.json').write_text(_json(receipt)+'\n')
    for private_file in (rrd, rbl, output/'EXPORT_RECEIPT.private.json'):
        private_file.chmod(0o600)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New private local output directory (must not exist)')
    p.add_argument('--episode', help='Optional exact configured episode id')
    p.add_argument('--allow-local-kinematics', action='store_true')
    args = p.parse_args()
    episodes = load_episodes(args.config, allow_local_kinematics=args.allow_local_kinematics)
    if args.episode:
        episodes = [e for e in episodes if e.id == args.episode]
        if not episodes:
            raise ValueError('CONFIGURED_EPISODE_NOT_FOUND')
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    receipts = [export_episode(e, args.output/e.id) for e in episodes]
    (args.output/'INDEX.private.json').write_text(_json({'episodes':receipts})+'\n')
    (args.output/'INDEX.private.json').chmod(0o600)
    print(_json({'status':'EXPORTED_PRIVATE_RERUN_FILES_NOT_NATIVE_SCIENTIFIC_QUALIFICATION',
                 'sdk_version':SDK_VERSION, 'episodes':len(receipts), 'RGB_records':sum(r['RGB_records'] for r in receipts),
                 'new_integrations':0, 'new_queries':0}))


if __name__ == '__main__':
    main()
