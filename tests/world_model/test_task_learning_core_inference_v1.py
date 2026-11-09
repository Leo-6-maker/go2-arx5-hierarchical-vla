"""CPU reconstruction fixtures do not grant Core GPU or native task admission."""
import pytest
from go2arx5_vla.world_model import task_learning_core_inference_v1 as task
from go2arx5_vla.world_model import target_gr00t_inference_v1 as shared


def test_memory_path_is_the_existing_qualified_official_logits_only_option():
    assert task.memory_implementation()==task.MEMORY_IMPLEMENTATION


def test_old_four_query_or_parent_shadow_identity_cannot_be_a_task_package():
    s=dict(profile=task.PROFILE,kind=task.KIND,execution_site='CORE_FUSION_NEURAL',
        device='cuda',physics=False,baseline_changed=False,max_queries=4,
        target_inference_mode=shared.PARENT_SHADOW)
    with pytest.raises(PermissionError,match='FIXED_CONTINUOUS_PACKAGE_PROBES'):
        shared.validate_spec(s)


def test_cross_site_source_is_explicit_and_keeps_legacy_and_hopper_names(tmp_path):
    import json
    import numpy as np
    rows=[dict(episode=str(i),split='TRAIN',anchor=0,seed=i,noise=dict(seed=i),
        model_global_step=0,model_identity=dict(dataset_sha256='a'*64,statistics_sha256='b'*64))for i in range(4)]
    a,b=tmp_path/'a',tmp_path/'b'
    for folder in(a,b):
        folder.mkdir();(folder/'RESULT.json').write_text(json.dumps(dict(records=rows)))
        for i in range(4):
            np.savez(folder/(str(i)+'.npz'),physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)))
    assert shared.compare_cross_site(a,b)['status'].startswith('PASS_ACTUAL_A40_CORE')
    r=shared.compare_cross_site(a,b,source_site='HOPPER_SINGLE_GPU')
    assert r['status'].startswith('PASS_ACTUAL_HOPPER_CORE')and r['actual_matched_queries']==4
    with pytest.raises(PermissionError,match='EXPLICIT_PAIRED_TRAINING_SITE'):
        shared.compare_cross_site(a,b,source_site='UNVERIFIED_SITE')
