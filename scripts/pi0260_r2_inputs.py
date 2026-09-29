"""Shared deterministic full-view processor and strict RGB/state/language boundary."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pi0260_r2_codec as codec

def build(statistics,cosmos):
    codec.install()
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    p=Gr00tN1d7Processor(modality_configs=codec.modality(),statistics=statistics,
        use_percentiles=True,clip_outliers=False,use_relative_action=True,
        image_crop_size=[256,256],image_target_size=[256,256],use_albumentations=False,
        random_rotation_angle=0,color_jitter_params=None,crop_fraction=1.,
        model_name=str(cosmos),model_type='qwen',max_state_dim=132,max_action_dim=132,max_action_horizon=40,
        state_dropout_prob=0.,exclude_state=False,transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True))
    # Same deterministic full field in train/eval, including no crop RNG consumer.
    p.train_image_transform=p.eval_image_transform
    p.set_statistics(statistics,override=True)
    assert p.embodiment_id_mapping['new_embodiment']==10
    return p

def batch(processor,packet,label=None):
    from gr00t.data.types import VLAStepData
    from gr00t.data.embodiment_tags import EmbodimentTag
    from PIL import Image
    state=np.asarray(packet['state_v1'],np.float32)
    assert state.shape==(10,) and np.isfinite(state).all()
    for stream in ('body_rgb','wrist_rgb'):
        assert packet[stream].shape==(96,96,3) and packet[stream].dtype==np.uint8
    assert isinstance(packet['language'],str)
    actions={} if label is None else dict(eef_9d=label[:,:9],gripper_position=label[:,9:])
    content=VLAStepData(images=dict(external_rgb=[Image.fromarray(packet['body_rgb'])],wrist_rgb=[Image.fromarray(packet['wrist_rgb'])]),
        states=dict(eef_9d=state[None,:9],gripper_position=state[None,9:]),actions=actions,masks=None,
        text=packet['language'],embodiment=EmbodimentTag.NEW_EMBODIMENT,metadata={})
    processor.eval() if label is None else processor.train()
    encoded=processor([dict(content=content)])
    result=processor.collator([encoded])
    return result

def inventory(value,prefix=''):
    import torch
    rows=[]
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous()
        return [dict(path=prefix,dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())]
    if isinstance(value,dict) or hasattr(value,'items'):
        for k,v in sorted(value.items()):rows+=inventory(v,prefix+'/'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):rows+=inventory(v,prefix+'/'+str(i))
    else:rows.append(dict(path=prefix,value=value))
    return rows

def identity(value):return hashlib.sha256(json.dumps(inventory(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
