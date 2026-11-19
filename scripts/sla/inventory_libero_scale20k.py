"""Build the SCALE20K LIBERO inventory from the pinned native LeRobot release."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from scripts.pi0292.durable import write


REPO = 'lerobot/libero'
REVISION = 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4'
BENCHMARK_REVISION = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
INFO_SHA = '630630a1a4f07fc9fc1f1b2b7987ed3bd034bd8cffaaf8a0e6939ae9c468445d'
EPISODE_SHA = '324db6d3f8efbfb340177507c11bfa2e78b1753c46802b3700fa8c8b4be461db'
TASK_SHA = '80bbbcdf623b7447c0f187af8ea70aa3a5c0018aaf56adba1731aeddcf18f5ff'
CAMERAS = ('observation.images.image', 'observation.images.image2')
FPS = 10
HORIZON = 10
INDEX_COLUMNS = ('episode_index', 'index', 'frame_index', 'task_index', 'timestamp')
# LIBERO Spatial is an entire held-out suite. Its task identity is checked
# against the pinned original BDDL filenames supplied in the release audit.
SOURCE_DEV_SUITE = 'libero_spatial'


def read_json(path):
    return json.loads(path.read_bytes())


def verified_index(path, keys):
    import pyarrow.parquet as parquet

    return parquet.read_table(path, columns=list(keys)).to_pydict()


def inventory(metadata, root, benchmark_tasks):
    import pyarrow.parquet as parquet

    info = metadata/'info.json'
    tasks = metadata/'tasks.parquet'
    episodes = metadata/'file-000.parquet'
    if (sha(info), sha(tasks), sha(episodes)) != (INFO_SHA, TASK_SHA, EPISODE_SHA):
        raise ValueError('LIBERO_PINNED_RELEASE_METADATA_BYTES_REQUIRED')
    schema = read_json(info)
    if (schema['total_episodes'], schema['total_tasks'], schema['total_frames'], schema['fps']) != (1693,40,273465,FPS):
        raise ValueError('LIBERO_PUBLISHED_NATIVE_CLOCK_AND_COUNTS_CHANGED')
    if (schema['splits'] != {'train':'0:1693'} or schema['features']['action']['shape'] != [7]
            or schema['features']['observation.state']['shape'] != [8]):
        raise ValueError('LIBERO_TRAIN_RELEASE_REQUIRES_NATIVE_STATE8_AND_ACTION7')

    names = {row['task_index']:row['__index_level_0__'] for row in parquet.read_table(tasks).to_pylist()}
    if set(names) != set(range(40)) or set(names.values()) != set(benchmark_tasks):
        raise ValueError('LIBERO_ALL_TASKS_REQUIRE_ORIGINAL_BDDL_IDENTITY')
    paths = read_json(metadata/'DATA_TREE.private.json')
    observed = {}
    for item in paths:
        path = root/item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('LIBERO_RAW_PARQUET_BYTES_CHANGED:'+item['path'])
        columns = verified_index(path, INDEX_COLUMNS)
        for episode, index, frame, task, timestamp in zip(*(columns[k] for k in INDEX_COLUMNS),strict=True):
            entry = observed.setdefault(episode, dict(start=index, rows=0, task=task, file=item['path']))
            if (index != entry['start']+frame or frame != entry['rows'] or task != entry['task']
                    or item['path'] != entry['file'] or not np.isclose(timestamp,frame/FPS,rtol=0,atol=1e-4)):
                raise ValueError('LIBERO_EPISODE_INDEX_TASK_OR_CLOCK_MISMATCH:'+str(episode))
            entry['rows'] += 1

    file_by_name = {row['path']:row for row in paths}
    video_by_name = {row['path']:row for row in read_json(metadata/'VIDEOS_TREE.private.json')}
    rows = []
    for raw in parquet.read_table(episodes).to_pylist():
        episode, length = raw['episode_index'],raw['length']
        from_index,to_index = raw['dataset_from_index'],raw['dataset_to_index']
        match = observed[episode]
        declared_file = schema['data_path'].format(chunk_index=raw['data/chunk_index'],file_index=raw['data/file_index'])
        if (match['start'],match['rows'],match['file']) != (from_index,length,declared_file) or to_index-from_index != length:
            raise ValueError('LIBERO_EPISODE_METADATA_DISAGREES_WITH_ROW_INDEX')
        task = names[match['task']]
        context = benchmark_tasks[task]
        cameras = {}
        for camera in CAMERAS:
            prefix = 'videos/'+camera+'/'
            name = schema['video_path'].format(video_key=camera,chunk_index=raw[prefix+'chunk_index'],
                                               file_index=raw[prefix+'file_index'])
            start,end = raw[prefix+'from_timestamp'],raw[prefix+'to_timestamp']
            if start < 0 or not np.isclose((end-start)*FPS,length,rtol=0,atol=1e-4):
                raise ValueError('LIBERO_METADATA_VIDEO_INTERVAL_MUST_MATCH_NATIVE_EPISODE')
            cameras[camera] = dict(**video_by_name[name],first_video_frame=round(start*FPS),video_frames=length)
        rows.append(dict(episode=episode,task_index=match['task'],task=task,suite=context['suite'],
                         scene_identity=context['scene_identity'],BDDL_file=context['file'],
                         upstream_split='train',split='SOURCE_DEV' if context['suite']==SOURCE_DEV_SUITE else 'TRAIN',
                         length=length,first_index=from_index,window_count=max(0,length-HORIZON+1),
                         data=file_by_name[declared_file],cameras=cameras))

    if (set(observed) != set(range(1693)) or [r['episode'] for r in rows] != list(range(1693))
            or sum(r['length'] for r in rows) != 273465 or len({r['task'] for r in rows}) != 40):
        raise ValueError('LIBERO_ORIGINAL_FULL_INDEX_RECONCILIATION_REQUIRED')
    sizes = Counter(row['split'] for row in rows)
    train = [row for row in rows if row['split']=='TRAIN']
    dev = [row for row in rows if row['split']=='SOURCE_DEV']
    if {r['task'] for r in train} & {r['task'] for r in dev}:
        raise ValueError('LIBERO_HELD_OUT_TASK_LEAKAGE')
    return dict(schema='SLA_SCALE20K_LIBERO_INDEX_V1',repo=REPO,revision=REVISION,benchmark_revision=BENCHMARK_REVISION,
        release_episode_count=len(rows),release_tasks=len(names),raw_frame_count=273465,planned_split_counts=dict(sizes),
        heldout_suite=SOURCE_DEV_SUITE,indexed_files=len(paths),total_indexed_bytes=sum(p['bytes'] for p in paths),
        horizon=HORIZON,native_fps=FPS,rows=rows,admitted_training_episodes=0,
        windows_policy='Contiguous native H10 commands within each original episode, no padding or state-to-action substitution',
        site_status='NUMERIC_INDEX_VERIFIED_RGB_AND_LABEL_AUDIT_REQUIRED',new_updates=0,new_model_generations=0,new_native_requests=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','data','benchmark_tasks','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    result = inventory(args.metadata,args.data,read_json(args.benchmark_tasks))
    write(args.output,result)


if __name__ == '__main__':
    main()
