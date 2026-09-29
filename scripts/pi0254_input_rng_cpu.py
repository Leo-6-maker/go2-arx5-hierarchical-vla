"""Read-only real image augmentation/RNG boundary check; no VLA/model import."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0254')
SOURCE=Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import pi0251_step_rng as rng
    doc=json.loads((ROOT/'source640/manifest.json').read_text());row=next(x for x in doc['files'] if x['episode_index']==1)
    path=ROOT/'source640'/row['path'];assert sha(path)==row['sha256']
    with np.load(path,allow_pickle=False) as z:image=z['body_rgb'][0].copy() if z['body_rgb'].ndim==4 else z['body_rgb'].copy()
    assert image.shape==(96,96,3) and image.dtype==np.uint8
    module_path=SOURCE/'gr00t/model/gr00t_n1d7/image_augmentations.py'
    spec=importlib.util.spec_from_file_location('pi0254_actual_image_augmentations',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((ROOT/'R576/training/experiment_cfg/final_processor_config.json').read_text())
    keys=('image_target_size','image_crop_size','random_rotation_angle','color_jitter_params','shortest_image_edge','crop_fraction','extra_augmentation_config','letter_box_transform')
    values={k:ast.literal_eval(config[k]) for k in keys}
    train,*_=module.build_image_transformations_albumentations(**values)
    rng.SEED=25402;rng.MAX_STEPS=4608;rng.configure_namespace('PI0254','SUPPORT576')
    rng.step_key_seed=lambda step:('PI0254_ORIGINAL_SHARED_VISIT_CPU',7888995677184945688)
    outer=torch.get_rng_state().clone()
    with rng.rng_context(1):a=train(image=image)['image']
    with rng.rng_context(1):b=train(image=image)['image']
    assert torch.equal(outer,torch.get_rng_state()) and not np.array_equal(a,b)
    h=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    result=dict(status='PASS_ACTUAL_STOCHASTIC_PREPROCESSING_ESCAPES_TORCH_ONLY_RNG_CPU',source_episode_id=row['source_episode_id'],source_tensor_sha256=row['sha256'],
        implementation_sha256=sha(module_path),config_sha256=sha(ROOT/'R576/training/experiment_cfg/final_processor_config.json'),transform_config=values,
        transformed_image_sha256=[h(a),h(b)],same_raw_image=True,same_Torch_key_seed=True,Torch_outer_restored=True,augmented_image_equal=False,
        actual_R_U_historical_leaf_reconstruction=False,causal_claim='Proves uncovered augmentation boundary on actual source/config. Does not identify the exact historical unequal leaf from two batch hashes.',
        runtime_training_modified=False,guard_bypassed=False,new_GPU_jobs=0,new_GR00T_loads=0,new_model_queries=0,new_backward=0,new_optimizer=0)
    (ROOT/'R576/input_rng_boundary_cpu_receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
