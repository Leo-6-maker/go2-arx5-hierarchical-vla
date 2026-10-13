"""Train-capable NPZ/PNG adapter, with no import of the legacy CPU guard."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

OFFICIAL_SHA='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
DOMAIN='FIXED_BASE_ARX5_TASK_SIM'
TAG='new_embodiment'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def official_import(source):
    source=Path(source)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=OFFICIAL_SHA:
        raise ValueError('OFFICIAL_SOURCE_SHA_MISMATCH')
    for name in ('gr00t/data/state_action/pose.py','gr00t/data/state_action/state_action_processor.py',
                 'gr00t/data/utils.py','gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py'):
        data=subprocess.check_output(['git','-C',str(source),'show',OFFICIAL_SHA+':'+name])
        if hashlib.sha256(data).hexdigest()!=sha(source/name):raise ValueError('OFFICIAL_SOURCE_DIRTY')
    sys.path.insert(0,str(source))


def modality():
    from gr00t.data.types import ModalityConfig,ActionConfig,ActionFormat,ActionType,ActionRepresentation
    return {TAG:{
        'video':ModalityConfig(delta_indices=[0],modality_keys=['external_rgb','wrist_rgb']),
        'state':ModalityConfig(delta_indices=[0],modality_keys=['eef_9d','gripper_position']),
        'action':ModalityConfig(delta_indices=list(range(40)),modality_keys=['eef_9d','gripper_position'],action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE,type=ActionType.EEF,format=ActionFormat.XYZ_ROT6D,state_key='eef_9d'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT,state_key='gripper_position')]),
        'language':ModalityConfig(delta_indices=[0],modality_keys=['annotation.language.task_description'])}}


def build_processor(source,cosmos,statistics):
    official_import(source)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    processor=Gr00tN1d7Processor(modality_configs=modality(),statistics=copy.deepcopy(statistics),
        use_percentiles=False,clip_outliers=False,use_relative_action=True,image_crop_size=[256,256],
        image_target_size=[256,256],use_albumentations=False,random_rotation_angle=0,color_jitter_params=None,
        crop_fraction=1.,model_name=str(cosmos),model_type='qwen',max_state_dim=132,max_action_dim=132,
        max_action_horizon=40,state_dropout_prob=0.,exclude_state=False,
        transformers_loading_kwargs={'local_files_only':True,'trust_remote_code':True})
    processor.train_image_transform=processor.eval_image_transform
    processor.eval()
    if processor.embodiment_id_mapping[TAG]!=10:raise ValueError('EMBODIMENT_ROUTE_CHANGED')
    return processor


class NativeDataset:
    def __init__(self,root,source,cosmos,split='TRAIN',*,for_optimization=False,raw_bindings=None):
        if split not in ('TRAIN','DEV'):raise ValueError('EXPLICIT_TRAIN_OR_DEV_REQUIRED')
        if for_optimization and split!='TRAIN':raise ValueError('DEV_MUST_NOT_ENTER_OPTIMIZER')
        self.root=Path(root)
        self.source,self.cosmos,self.split=str(source),str(cosmos),split
        self.for_optimization=for_optimization
        self.manifest=read(self.root/'DATASET_MANIFEST.json')
        if self.manifest['domain']!=DOMAIN or self.manifest.get('purpose')=='NOT_ADMITTED_COMMISSIONING_DIAGNOSTIC':
            raise ValueError('WRONG_DATA_DOMAIN')
        self.statistics_sha=self.manifest['statistics_sha256']
        self.statistics=self.get_dataset_statistics()
        self.manifest_sha=sha(self.root/'DATASET_MANIFEST.json')
        self.rows=[(row,anchor) for row in self.manifest['episodes'] if row['split']==split for anchor in range(row['windows'])]
        if not self.rows:raise ValueError('EMPTY_SPLIT_NOT_QUALIFIED')
        self.raw_bindings=raw_bindings or {}
        self.processor=None
        self.cached_id,self.cached=None,None
        self.seed=29200

    def __len__(self):return len(self.rows)

    def get_dataset_statistics(self):
        if sha(self.root/'statistics.json')!=self.statistics_sha:raise ValueError('FROZEN_STATISTICS_CHANGED')
        return read(self.root/'statistics.json')

    def reset_seed(self,seed):self.seed=int(seed)

    def __getstate__(self):
        state=self.__dict__.copy()
        state.update(processor=None,cached_id=None,cached=None)
        return state

    def initialize(self):
        if self.processor is None:self.processor=build_processor(self.source,self.cosmos,self.get_dataset_statistics())

    def raw(self,index):
        row,anchor=self.rows[index]
        path=self.root/row['file']
        if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('DATASET_PATH_ESCAPE')
        if self.cached_id!=row['id']:
            if sha(path)!=row['sha256']:raise ValueError('NUMERIC_EXPORT_CHANGED')
            with np.load(path,allow_pickle=False) as archive:self.cached={k:archive[k].copy() for k in archive.files}
            self.cached_id=row['id']
        packet=self.cached
        if not packet['valid_window'][anchor] or packet['label_representation'].item()!='ABSOLUTE_COMMITTED_EEF10':
            raise ValueError('NONCOMMITTED_OR_RELATIVE_LABEL')
        rawroot=Path(self.raw_bindings.get(row['id'],row['raw_root'])).resolve()
        images=[]
        for camera in range(2):
            frame=(rawroot/packet['rgb_paths'][anchor,camera]).resolve()
            if not frame.is_relative_to(rawroot) or sha(frame)!=packet['rgb_sha256'][anchor,camera]:
                raise ValueError('RAW_RGB_CHANGED_OR_PATH_ESCAPE')
            with Image.open(frame) as image:
                if image.mode!='RGB' or image.size!=(256,256):raise ValueError('RAW_RGB_FORMAT_CHANGED')
                images.append(np.asarray(image).copy())
        return packet['actual_state10'][anchor],packet['committed_absolute10'][anchor:anchor+40],images,packet['instruction'].item()

    def identity(self,index):
        row,anchor=self.rows[index]
        return dict(episode_id=row['id'],split=self.split,anchor=anchor)

    def __getitem__(self,index):
        self.initialize()
        from gr00t.data.types import VLAStepData,MessageType
        from gr00t.data.embodiment_tags import EmbodimentTag
        state,commands,images,text=self.raw(index)
        value=VLAStepData(images={'external_rgb':[images[0]],'wrist_rgb':[images[1]]},
            states={'eef_9d':state[None,:9],'gripper_position':state[None,9:]},
            actions={'eef_9d':commands[:,:9],'gripper_position':commands[:,9:]},masks=None,text=text,
            embodiment=EmbodimentTag.NEW_EMBODIMENT,metadata={})
        return {'processed':self.processor([{'type':MessageType.EPISODE_STEP.value,'content':value}])}


class ProcessedCollator:
    def __init__(self,processor):self.collator=processor.collator

    def __call__(self,rows):
        if any(set(row)!={'processed'} for row in rows):raise ValueError('AUDIT_IDENTITY_IN_TRAIN_BATCH')
        return self.collator([row['processed'] for row in rows])


class CyclicSampler:
    """Cursor is the consumed sample count, never worker-prefetch progress."""
    def __init__(self,size,seed=29200,start=0,stop=131072):
        if size<=0 or not 0<=start<=stop:raise ValueError('INVALID_SAMPLER_RANGE')
        self.size,self.seed,self.start,self.stop=int(size),int(seed),int(start),int(stop)

    def __len__(self):return self.stop-self.start

    def __iter__(self):
        epoch=-1
        for cursor in range(self.start,self.stop):
            cycle,offset=divmod(cursor,self.size)
            if cycle!=epoch:
                permutation=np.random.Generator(np.random.PCG64(self.seed+cycle)).permutation(self.size)
                epoch=cycle
            yield int(permutation[offset])

    def state_dict(self,consumed):
        if not self.start<=consumed<=self.stop:raise ValueError('INVALID_CONSUMED_CURSOR')
        return dict(size=self.size,seed=self.seed,cursor=consumed,stop=self.stop,
                    algorithm='NUMPY_PCG64_SEED_PLUS_CYCLE_PERMUTATION_V1')
