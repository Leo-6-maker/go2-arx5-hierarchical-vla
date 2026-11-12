"""One frozen OmniVGGT cache pass; no SLA model, optimization or physics."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def pinned(pin):
    if sha(pin['path'])!=pin['sha256']:
        raise ValueError('FROZEN_TEACHER_INPUT_PIN_CHANGED')
    return json.loads(Path(pin['path']).read_bytes())


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().numpy()
        h.update((name+':'+str(a.shape)+':'+str(a.dtype)+'\n').encode());h.update(a.tobytes())
        del a
    return h.hexdigest()


def run(spec,counts):
    import torch
    from safetensors.torch import load_file
    from omnivggt.models.omnivggt import OmniVGGT
    from vggt.utils.load_fn import preprocess_images_from_openpi
    from vggt.heads.utils import custom_pooling
    import torch.nn.functional as F
    if torch.cuda.device_count()!=1:
        raise ValueError('ONE_ACTUAL_FROZEN_TEACHER_GPU_REQUIRED')
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.manual_seed(20261112);torch.cuda.manual_seed_all(20261112)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    assets=pinned(spec['teacher_manifest'])
    if assets['revision']!='4a0917605798f1e8a645153ab8cdfb12aa453354' or not assets['all_files_verified']:
        raise ValueError('PINNED_OMNIVGGT_COMPONENT_REQUIRED')
    weight=Path(spec['teacher_weight'])
    if sha(weight)!='c9c3772b9bbfc648fa95ebffb3d9ff856f21e9b9712685eb0f98add53897969d':
        raise ValueError('EXACT_TEACHER_WEIGHT_BYTES_REQUIRED')
    pixel_root=Path(spec['pixel_view']['path'])
    if sha(pixel_root/'MANIFEST.private.json')!=spec['pixel_view']['sha256']:
        raise ValueError('EXACT_STUDENT_RGB_VIEW_MANIFEST_REQUIRED')
    view=json.loads((pixel_root/'MANIFEST.private.json').read_bytes())
    if view['schema']!='SLA_EXACT_TRAIN_RGB224_TEACHER_VIEW_V1' or not view['passes'] or len(view['rows'])!=1450 or view['privileged_inputs_saved']:
        raise ValueError('ALL_TRAIN_ONLY_CURRENT_STUDENT_PIXELS_REQUIRED')
    model=OmniVGGT(enable_camera=False,enable_point=False,enable_depth=False)
    incompatible=model.load_state_dict(load_file(str(weight),device='cpu'),strict=False)
    if incompatible.missing_keys or any(not k.startswith(('camera_head.','point_head.','depth_head.')) for k in incompatible.unexpected_keys):
        raise ValueError('TEACHER_AGGREGATOR_NOT_FULLY_RESTORED:'+str(incompatible))
    model.eval().requires_grad_(False).to('cuda')
    torch.cuda.synchronize();counts['teacher_model_loads']=1
    initial=fingerprint(model)
    root=Path(spec['registered_root'])/'features';root.mkdir(exist_ok=False)
    rows=[];repeat_exact=None
    def extract(pixels,padding):
        x=pixels.astype(np.float32)/255.0*2.0-1.0
        images=[]
        for i in range(2):
            image=torch.as_tensor(x[i:i+1],device='cuda').float()
            image=torch.clamp(image/2.0+.5,0,1)
            image[~torch.as_tensor(padding[i:i+1],device='cuda')]=1.
            images.append(image.permute(0,3,1,2).contiguous())
        counts['teacher_forward_attempts']+=1
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            prepared=preprocess_images_from_openpi(images)
            features,start=model.aggregator(prepared,feature_layer=-1)
            if len(features)!=1:
                raise ValueError('ONLY_FINAL_FROZEN_TEACHER_LAYER_REQUIRED')
            hidden=features[0][:,:,start:,:]
            h,w=prepared.shape[-2:]
            reference=torch.zeros((1,512,1),dtype=hidden.dtype,device='cuda')
            pooled=custom_pooling(hidden,(h//14,w//14),(h,w),reference,'bilinear',False)
            target=pooled.float().to(torch.bfloat16).contiguous()
        mask=F.interpolate(torch.as_tensor(padding[None],device='cuda').float(),size=(16,16),mode='nearest').bool().flatten(start_dim=1)
        torch.cuda.synchronize();counts['teacher_forwards_completed']+=1
        if target.shape!=(1,512,2048) or not torch.isfinite(target).all():
            raise ValueError('COMPLETE_FINITE_TWO_VIEW_TEACHER_TARGET_REQUIRED')
        bits=target[0].cpu().view(torch.int16).numpy().view(np.uint16).copy()
        return bits,mask[0].cpu().numpy().copy()
    for row in view['rows']:
        if time.time()>=spec['deadline_epoch']:
            raise TimeoutError('FINITE_FROZEN_TEACHER_CACHE')
        path=pixel_root/row['file']
        if sha(path)!=row['sha256']:
            raise ValueError('SAVED_STUDENT_PIXELS_CHANGED')
        with np.load(path,allow_pickle=False) as z:
            pixels=z['pixels'];padding=z['padding_mask']
        if hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixels_sha256']:
            raise ValueError('SAME_STUDENT_PIXELS_REQUIRED')
        bits,mask=extract(pixels,padding)
        if row['index']==0:
            other,other_mask=extract(pixels,padding)
            repeat_exact=np.array_equal(bits,other) and np.array_equal(mask,other_mask)
            if not repeat_exact:
                raise ValueError('FROZEN_TEACHER_REPEAT_NOT_BIT_EXACT')
        file=root/f"{row['index']:04d}.npz"
        np.savez(file,bf16_bits=bits,mask=mask)
        rows.append(dict(index=row['index'],episode_id=row['episode_id'],anchor=row['anchor'],file=file.name,
                         sha256=sha(file),pixels_sha256=row['pixels_sha256'],padding_mask_sha256=row['padding_mask_sha256']))
        counts['complete_cached_TRAIN_windows']+=1
        if len(rows)%25==0:
            write(Path(spec['registered_root'])/'PROGRESS.private.json',dict(counts,teacher_updates=0,SLA_updates=0,physical_requests=0))
    after=fingerprint(model)
    if initial!=after or any(p.requires_grad for p in model.parameters()):
        raise ValueError('FROZEN_TEACHER_PARAMETERS_CHANGED')
    value=dict(schema='SLA_FROZEN_OMNIVGGT_TARGET_CACHE_V1',passes=True,rows=rows,
        source_dataset_sha256=view['source_dataset_sha256'],target_audit=view['target_audit'],
        source_pixel_view=spec['pixel_view'],teacher_manifest=spec['teacher_manifest'],
        teacher_weight_sha256=sha(weight),source_camera_count=2,teacher_last_layer=-1,
        teacher_gradient=False,privileged_inputs_used=False,missing_third_camera_fed_to_teacher=False,
        pooling='bilinear',use_vggt_pe=False,target_storage='EXACT_BFLOAT16_BITS_AS_UINT16',
        teacher_frozen_fingerprint=initial,teacher_repeat_bit_exact=repeat_exact,
        inherited_checkpoint_unexpected_head_keys=incompatible.unexpected_keys,
        actual=counts,teacher_updates=0,SLA_updates=0,physical_requests=0)
    write(root/'MANIFEST.private.json',value)
    return dict(schema='SLA_FROZEN_TEACHER_CACHE_ACTUAL_V1',passes=True,
        teacher_cache=dict(path=str(root),sha256=sha(root/'MANIFEST.private.json')),
        actual=counts,frozen_fingerprint_unchanged=initial,teacher_repeat_bit_exact=repeat_exact,
        teacher_updates=0,SLA_updates=0,physical_requests=0,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())


if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    counts=dict(teacher_model_loads=0,teacher_forward_attempts=0,teacher_forwards_completed=0,complete_cached_TRAIN_windows=0)
    try:
        result=run(spec,counts)
    except BaseException as e:
        write(Path(spec['registered_root'])/'FAILED.private.json',dict(error=repr(e),actual=counts,teacher_updates=0,SLA_updates=0,physical_requests=0))
        raise
    write(Path(spec['registered_root'])/'RESULT.private.json',result)
