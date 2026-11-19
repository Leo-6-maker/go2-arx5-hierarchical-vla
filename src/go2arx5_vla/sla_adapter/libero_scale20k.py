"""LIBERO native H10 data. No physical action conversion or terminal padding."""
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .runtime_v1 import read_pin


FPS, HORIZON = 10, 10
VIEWS = {'base_0_rgb':'observation.images.image', 'left_wrist_0_rgb':'observation.images.image2'}
NUMERIC_FIELDS = ('observation.state','action','episode_index','index','frame_index','task_index','timestamp')


def numeric_episode(path, row):
    import pyarrow.parquet as parquet
    import pyarrow.compute as pc

    data = parquet.read_table(path,columns=list(NUMERIC_FIELDS))
    data = data.filter(pc.equal(data['episode_index'],row['episode']))
    fields = data.to_pydict()
    state,action = (np.asarray(fields[key],dtype=np.float32) for key in NUMERIC_FIELDS[:2])
    length = row['length']
    if (state.shape != (length,8) or action.shape != (length,7)
            or fields['frame_index'] != list(range(length))
            or fields['index'] != list(range(row['first_index'],row['first_index']+length))
            or set(fields['task_index']) != {row['task_index']}
            or not np.allclose(fields['timestamp'],np.arange(length)/FPS,rtol=0,atol=1e-4)
            or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ValueError('LIBERO_EXACT_FINITE_NATIVE_EPISODE_REQUIRED')
    return state,action


def camera_image(path, asset, anchor):
    import av

    index = asset['first_video_frame']+anchor
    with av.open(str(path),mode='r') as container:
        stream, = container.streams.video
        if (stream.average_rate != FPS or stream.start_time != 0
                or stream.frames < asset['first_video_frame']+asset['video_frames']):
            raise ValueError('LIBERO_NATIVE_VIDEO_STREAM_INVALID')
        stream.codec_context.thread_count = 1
        target = Fraction(index,FPS)/stream.time_base
        if target.denominator != 1:
            raise ValueError('LIBERO_FRAME_MUST_MAP_TO_AN_EXACT_VIDEO_PTS')
        container.seek(int(target),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts > target:
                break
            if frame.pts == target:
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape == (256,256,3) and rgb.dtype == np.uint8:
                    return rgb
                break
    raise ValueError('LIBERO_CURRENT_RGB_FRAME_IS_NOT_EXACTLY_ALIGNED')


def normalize(value, quantiles):
    lo,hi = (np.asarray(quantiles[key],np.float32) for key in ('q01','q99'))
    if lo.shape != (value.shape[-1],) or hi.shape != lo.shape or np.any(hi < lo) or not np.isfinite([lo,hi]).all():
        raise ValueError('LIBERO_TRAIN_ONLY_TYPED_QUANTILE_RANGE_REQUIRED')
    return (value-lo)/(hi-lo+1e-6)*2-1


def pad32(value):
    return np.pad(value,[(0,0)]*(value.ndim-1)+[(0,32-value.shape[-1])])


class LiberoDataset:
    def __init__(self,root,manifest,*,split,resize_with_pad,max_frames_cached=128):
        self.root = Path(root).resolve(strict=True)
        self.manifest = read_pin(manifest)
        self.resize = resize_with_pad
        self.max_frames_cached = max_frames_cached
        if self.manifest['schema'] != 'SLA_SCALE20K_LIBERO_ADMITTED_V1' or split not in ('TRAIN','SOURCE_DEV'):
            raise ValueError('LIBERO_ADMITTED_TRAIN_OR_DEV_MANIFEST_REQUIRED')
        self.statistics = self.manifest['train_statistics']
        self.episodes = {r['episode']:r for r in self.manifest['rows'] if r['split']==split and r['admitted']}
        self.rows = [(episode,k) for episode,r in self.episodes.items() for k in range(r['window_count'])]
        self.verified,self.numeric,self.images = {},OrderedDict(),OrderedDict()

    def __len__(self):
        return len(self.rows)

    def identity(self,index):
        episode,anchor = self.rows[index]
        r = self.episodes[episode]
        return dict(episode=episode,anchor=anchor,supervised_indices=[anchor,anchor+HORIZON-1],native_fps=FPS,
                    task=r['task'],suite=r['suite'],current_observation_index=anchor)

    def path(self,asset):
        name = asset['path']
        if name not in self.verified:
            path = (self.root/name).resolve(strict=True)
            if not path.is_relative_to(self.root) or path.stat().st_size != asset['bytes'] or sha(path) != asset['sha256']:
                raise ValueError('LIBERO_PINNED_FILE_BYTES_REQUIRED:'+name)
            self.verified[name] = path
        return self.verified[name]

    def __getitem__(self,index):
        episode,anchor = self.rows[index]; row = self.episodes[episode]
        if episode not in self.numeric:
            self.numeric[episode] = numeric_episode(self.path(row['data']),row)
            if len(self.numeric) > 32:
                self.numeric.popitem(last=False)
        self.numeric.move_to_end(episode)
        state,commands = self.numeric[episode]
        action = commands[anchor:anchor+HORIZON]
        if action.shape != (HORIZON,7):
            raise ValueError('LIBERO_NATIVE_H10_MUST_NOT_COPY_TERMINAL_ACTIONS')
        key = (episode,anchor)
        if key not in self.images:
            images,pads = {},{}
            for view,camera in VIEWS.items():
                asset = row['cameras'][camera]
                rgb = camera_image(self.path(asset),asset,anchor)
                images[view],pads[view] = self.resize(rgb,224,224,return_mask=True)
            # Pi0's usual three camera slots: physically absent right wrist.
            images['right_wrist_0_rgb'] = np.zeros_like(images['left_wrist_0_rgb'])
            pads['right_wrist_0_rgb'] = np.ones_like(pads['left_wrist_0_rgb'],dtype=bool)
            self.images[key] = images,pads
            if len(self.images) > self.max_frames_cached:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        images,pads = self.images[key]
        return dict(image=images,image_padding_mask=pads,image_mask={view:view in VIEWS for view in images},
                    state=pad32(normalize(state[anchor],self.statistics['state'])),
                    actions=pad32(normalize(action,self.statistics['action'])),prompt=row['task'])
