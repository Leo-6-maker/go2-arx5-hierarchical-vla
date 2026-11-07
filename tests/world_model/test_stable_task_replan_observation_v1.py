"""Synthetic sealed RGB packets verify fresh clocks and explicit feature fields."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scripts.pi0292.relay_protocol import sealed
from go2arx5_vla.world_model import stable_task_dataset_v1 as dataset
from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
from go2arx5_vla.world_model.storage import digest


def test_next_observation_cannot_be_replaced_by_stale_first_packet(tmp_path):
    root=tmp_path/'world';folder=root/'native-01/PAUSED_OBSERVATIONS/synthetic';(folder/'images').mkdir(parents=True)
    model=root/'scene';model.write_bytes(b'not a native model; input parser fixture only')
    step=37303;pin='a'*64;state=folder/'STATE.private.bin';state.write_bytes(b'synthetic-state')
    image=np.zeros((256,256,3),np.uint8);image[90:110,100:120,0]=200
    for slot in ('body_rgb','wrist_rgb'):Image.fromarray(image).save(folder/'images'/f'000000-{slot}.png')
    files={p.relative_to(folder).as_posix():digest(p)for p in [state,*sorted((folder/'images').glob('*.png'))]}
    shadow=sealed(dict(native_step=step,target_observation=dict(target_runtime_sha256=pin,
        native_state_sha256=digest(state),actual_state_eef10=[0.,0.,0.,1.,0.,0.,0.,1.,0.,.95]),
        object_truth_forbidden_student_input=[99.,99.,99.]))
    body=dict(native_step=step,world=dict(execution_spec_sha256=pin,scene_sha256=digest(model),native_output=str(root/'native-01')),
        files=files,original_shadow_ABI=shadow,future_achieved_state_forbidden=[42.]*10)
    path=folder/'PAUSED_OBSERVATION.private.json';path.write_text(json.dumps(sealed(body)))
    observation=dict(path=str(path),sha256=digest(path),native_step=step)
    spec=dict(registered_root=str(root),model_path=str(model),phase=prefix.PHASE,kind=prefix.KIND,
        stable_task_closed_loop=prefix.protocol(),live_skill_start_step=prefix.B,max_ticks=prefix.E,
        WBC_changes=False,physics_dt_s=.002,task_command='APPROACH_RED_4MM_OPEN')
    x,_=dataset.actual_features(root,spec,pin,dict(student_observation=observation),expected_step=step)
    assert x.shape==(116,)and np.isfinite(x).all()
    np.testing.assert_allclose(x[-12:-2],[0.,0.,0.,1.,0.,0.,0.,1.,0.,.95])
    with pytest.raises(ValueError,match='SAME_REAL_WORLD_SPEC_AND_DECISION'):
        dataset.actual_features(root,spec,pin,dict(student_observation=observation),expected_step=40303)
    with pytest.raises(ValueError,match='PREREGISTERED_REAL_REPLAN'):
        dataset.actual_features(root,spec,pin,dict(student_observation=observation),expected_step=step+25)
    with pytest.raises(ValueError,match='EXACT_NEW_TYPED'):
        dataset.actual_features(root,dict(spec,phase='collect-target-stable-task-eval-v1'),pin,
            dict(student_observation=observation),expected_step=step)
