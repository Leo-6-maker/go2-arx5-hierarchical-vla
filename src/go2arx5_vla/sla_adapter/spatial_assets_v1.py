"""Frozen spatial targets and inherited projector, outside deployment inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .checkpoint_v1 import sha
from .runtime_v1 import read_pin


def load_projector(pin, parent_manifest, recipe):
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import jax
    from types import SimpleNamespace
    from openpi.models.projectors import AlignProjector
    root=Path(pin['path'])
    if sha(root/'MANIFEST.private.json') != pin['sha256']:
        raise ValueError('INHERITED_PROJECTOR_MANIFEST_CHANGED')
    m=json.loads((root/'MANIFEST.private.json').read_bytes())
    if m['schema']!='SLA_INHERITED_ALIGN_PROJECTOR_ASSET_V1' or m['parent_manifest']!=parent_manifest or m['random_initialization']:
        raise ValueError('SAME_BASE_INHERITED_PROJECTOR_REQUIRED')
    cfg=SimpleNamespace(**recipe)
    template=nnx.eval_shape(lambda:AlignProjector(2048,cfg,nnx.Rngs(20261112)))
    graph,state=nnx.split(template)
    expected=flatten_dict(state.to_pure_dict())
    values={}
    for row in m['arrays']:
        p=root/row['file'];path=tuple(row['path'])
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=row['sha256']:
            raise ValueError('INHERITED_PROJECTOR_BYTES_CHANGED')
        a=np.load(p,allow_pickle=False)
        if path not in expected or a.shape!=expected[path].shape or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('INHERITED_PROJECTOR_TEMPLATE_MISMATCH')
        values[path]=jax.device_put(a)
    if set(values)!=set(expected) or len(values)!=6:
        raise ValueError('ALL_SIX_INHERITED_PROJECTOR_LEAVES_REQUIRED')
    state.replace_by_pure_dict(unflatten_dict(values))
    return graph,state,m


class FeatureBank:
    def __init__(self,pin,*,source_dataset_sha256,target_audit_sha256):
        self.root=Path(pin['path'])
        if sha(self.root/'MANIFEST.private.json')!=pin['sha256']:
            raise ValueError('IMMUTABLE_SPATIAL_CACHE_MANIFEST_CHANGED')
        self.manifest=json.loads((self.root/'MANIFEST.private.json').read_bytes())
        m=self.manifest
        if (m['schema']!='SLA_FROZEN_OMNIVGGT_TARGET_CACHE_V1' or not m['passes']
            or m['source_dataset_sha256']!=source_dataset_sha256
            or m['target_audit']['sha256']!=target_audit_sha256
            or len(m['rows'])!=1450 or m['teacher_gradient'] or m['privileged_inputs_used']
            or m['source_camera_count']!=2 or m['teacher_last_layer']!=-1):
            raise ValueError('COMPLETE_MATCHED_TRAIN_ONLY_FROZEN_TEACHER_CACHE_REQUIRED')
        self.rows=m['rows']
        self.reads=0

    def for_index(self,index,item):
        import jax
        import ml_dtypes
        row=self.rows[index]
        pixels=np.stack([item['image'][k] for k in ('base_0_rgb','left_wrist_0_rgb')])
        padding=np.stack([item['image_padding_mask'][k] for k in ('base_0_rgb','left_wrist_0_rgb')])
        if (row['index']!=index or hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixels_sha256']
            or hashlib.sha256(padding.tobytes()).hexdigest()!=row['padding_mask_sha256']
            or item['image_mask']['right_wrist_0_rgb']):
            raise ValueError('SPATIAL_TARGET_MUST_MATCH_EXACT_CURRENT_STUDENT_PIXELS_AND_MASKS')
        path=self.root/row['file']
        if not path.resolve().is_relative_to(self.root.resolve()) or sha(path)!=row['sha256']:
            raise ValueError('SPATIAL_TARGET_CACHE_BYTES_CHANGED')
        with np.load(path,allow_pickle=False) as z:
            bits=z['bf16_bits'];mask=z['mask']
            if bits.shape!=(512,2048) or bits.dtype!=np.uint16 or mask.shape!=(512,) or mask.dtype!=bool:
                raise ValueError('EXACT_BFLOAT16_SPATIAL_512_VALID_VIEW_TOKENS_REQUIRED')
            features=bits.view(ml_dtypes.bfloat16)
            if not np.isfinite(features).all():
                raise ValueError('FINITE_FROZEN_TEACHER_TARGET_REQUIRED')
            target=np.zeros((768,2048),dtype=ml_dtypes.bfloat16);target[:512]=features
            valid=np.zeros(768,dtype=bool);valid[:512]=mask
        self.reads+=1
        return jax.device_put(target[None]),jax.device_put(valid[None])
