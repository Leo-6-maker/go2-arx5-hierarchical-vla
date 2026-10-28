#!/usr/bin/env python3
"""Existing Stage21 worker, with observational S4 noise and head-call receipts."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0252')
CODE = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
STAGE10 = Path('/home/liuyu-nfs/Projects/isaac-gr00t')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HOME='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/hf_home')
sys.path[:0] = [str(CODE / 'scripts'), str(CODE / 'src'), str(STAGE10)]
import numpy as np
import torch
import stage21_same_scene_groot_inference as inference

original_processor = inference._build_processor
def eval_processor(*args, **kwargs):
    processor, embodiment = original_processor(*args, **kwargs)
    processor.eval()
    return processor, embodiment
inference._build_processor = eval_processor

original_run = inference._run_one_raw
def observed_run(model, processor, embodiment, observation, **kwargs):
    assert model.action_head.num_inference_timesteps == 4
    assert not model.training
    noise, head_calls = [], []
    original_randn = torch.randn
    def observed_randn(*args, **kw):
        value = original_randn(*args, **kw)
        if tuple(value.shape) == (1, 40, 132):
            noise.append({'raw_bytes': value.detach().contiguous().view(torch.uint8).cpu().numpy().copy(),
                          'float_values': value.detach().float().cpu().numpy().copy(),
                          'dtype': str(value.dtype), 'device': str(value.device), 'shape': list(value.shape)})
        return value
    hook = model.action_head.model.register_forward_hook(lambda module, args, out: head_calls.append(1))
    torch.randn = observed_randn
    result = None
    error = None
    try:
        result = original_run(model, processor, embodiment, observation, **kwargs)
        assert len(noise) == 1 and len(head_calls) == 4, (len(noise), len(head_calls))
        return result
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        torch.randn = original_randn
        hook.remove()
        mailbox = Path(observation['path']).parent.parent
        stem = Path(observation['path']).stem
        artifact = mailbox / 'responses' / (stem + '_initial_noise.npz')
        metadata = []
        if noise:
            assert len(noise) == 1
            item = noise[0]
            np.savez_compressed(artifact, raw_bytes=item['raw_bytes'], float_values=item['float_values'])
            metadata.append({k: v for k, v in item.items() if k not in ('raw_bytes', 'float_values')})
            metadata[0]['raw_tensor_sha256'] = hashlib.sha256(item['raw_bytes'].tobytes()).hexdigest()
        receipt = {'schema': 'PI0252_ACTUAL_S4_QUERY_RECEIPT_V1',
                   'status': 'PASS' if result is not None and error is None else 'ENGINEERING_INCOMPLETE',
                   'query_seed': kwargs['seed'], 'head_calls': len(head_calls), 'noise_count': len(noise),
                   'noise': metadata, 'noise_artifact': str(artifact),
                   'pure_noise': True, 'rtc': False, 'target_in_model_input': False,
                   'model_mode_after': 'eval' if not model.training else 'train',
                   'error': error, 'forward_attempt_count': 1}
        (mailbox / 'responses' / (stem + '_sampler.json')).write_text(json.dumps(receipt, indent=2)+'\n')
inference._run_one_raw = observed_run

if __name__ == '__main__':
    from stage21_same_scene_groot_worker import main
    raise SystemExit(main())
