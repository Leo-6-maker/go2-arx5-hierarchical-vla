"""Qualify LIBERO numeric labels and the two current RGB streams for SCALE20K."""
import argparse
from collections import Counter
import hashlib
import time
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.libero_scale20k import HORIZON, camera_image, numeric_episode
from scripts.pi0292.durable import read, write
from scripts.sla.inventory_libero_scale20k import REPO, REVISION


def admit(index, root, *, pilot):
    import av
    if (index['schema'] != 'SLA_SCALE20K_LIBERO_INDEX_V1' or index['repo'] != REPO or index['revision'] != REVISION
            or index['heldout_suite'] != 'libero_spatial' or index['horizon'] != HORIZON):
        raise ValueError('LIBERO_EXACT_PINNED_PRE_SPLIT_INDEX_REQUIRED')
    rows = index['rows']
    if pilot:
        rows = [next(row for row in rows if row['split']==split) for split in ('TRAIN','SOURCE_DEV')]
    result = []
    verified_files = {}
    duplicates = {}
    split_counts = Counter()
    train = {'state':[],'action':[]}

    def locate(item):
        name = item['path']
        if name not in verified_files:
            path = root/name
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('LIBERO_PINNED_RAW_FILE_BYTES_REQUIRED:'+name)
            verified_files[name] = path
        return verified_files[name]

    for row in rows:
        state,action = numeric_episode(locate(row['data']),row)
        last_window = len(action)-HORIZON
        if last_window < 0 or row['window_count'] != last_window+1:
            raise ValueError('LIBERO_VALID_H10_WINDOW_COUNT_MUST_MATCH_NATIVE_LENGTH')
        fingerprint = hashlib.sha256(state.tobytes()+action.tobytes()).hexdigest()
        if fingerprint in duplicates:
            raise ValueError(f'LIBERO_DUPLICATE_NUMERIC_EPISODE:{duplicates[fingerprint]}:{row["episode"]}')
        duplicates[fingerprint] = row['episode']
        checks = []
        for camera,asset in row['cameras'].items():
            video = locate(asset)
            for at in (0,last_window):
                rgb = camera_image(video,asset,at)
                checks.append(dict(camera=camera,current_observation_index=at,
                    video_frame=asset['first_video_frame']+at,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        split_counts[row['split']] += 1
        if row['split']=='TRAIN':
            train['state'].append(state[:row['window_count']])
            train['action'].append(action)
        result.append(dict(row,admitted=True,numeric_state_action_sha256=fingerprint,verified_current_RGB=checks,
                           maximum_absolute_native_action=float(np.max(np.abs(action))),
                           native_gripper_values=np.unique(action[:,6]).tolist()))

    stats = {}
    for field,blocks in train.items():
        values = np.concatenate(blocks,axis=0)
        stats[field] = dict(samples=len(values),q01=np.quantile(values,.01,axis=0).tolist(),q99=np.quantile(values,.99,axis=0).tolist())
    return dict(schema='SLA_SCALE20K_LIBERO_ADMITTED_V1',repo=REPO,revision=REVISION,heldout_suite=index['heldout_suite'],
        rows=result,train_statistics=stats,admitted_episodes=dict(split_counts),verified_file_count=len(verified_files),
        verified_decoded_frames=sum(len(row['verified_current_RGB']) for row in result),
        active_input_dim=8,active_action_dim=7,native_fps=10,horizon=HORIZON,command_scaling='native OSC action, unchanged',
        video_decoder='PyAV '+av.__version__+' CPU exact stream PTS',
        collection_lineage='Pinned LeRobot release and original LIBERO BDDL suite/task. Per-episode original scene reset UUID is unavailable.',
        pilot=pilot,new_updates=0,new_model_generations=0,new_native_requests=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index','root','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    result = admit(read(args.index),args.root,pilot=args.pilot)
    write(args.output,dict(result,index_sha256=sha(args.index),wall_s=time.monotonic()-started))


if __name__ == '__main__':
    main()
