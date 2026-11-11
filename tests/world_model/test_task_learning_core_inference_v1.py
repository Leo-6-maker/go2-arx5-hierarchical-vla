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


def test_full_category_gpu_route_requires_its_own_terminal_memory_proof(tmp_path):
    import json
    from go2arx5_vla.world_model.storage import digest
    p=tmp_path/'memory.json'
    old=dict(schema='GR00T_CORE_FRESH_ORIGINAL_RECONSTRUCTION_ACTUAL_V1',source_sha='a'*40,
        worker_exit_code=0,lease_status='CLOSED',all_owned_children_terminal=True,
        proof=dict(all132_H40_outputs_bit_exact=True,physical_outputs_bit_exact=True,precision_relaxed=False))
    p.write_text(json.dumps(old));s=dict(memory_execution=task.MEMORY,checkpoint_parameter_bytes=12576064000,
        allocator_limit_bytes=13958643712,memory_execution_reference=dict(path=str(p),sha256=digest(p)))
    with pytest.raises(PermissionError,match='NOT_CPU_TABLE_QUALIFICATION'):task.validate_memory(s)
    actual=dict(old,schema='GR00T_CORE_ORDINARY_FROZEN_PARENT_PLUS_DELTA_ACTUAL_V1',proof=dict(old['proof'],
        status='PASS_ACTUAL_ORDINARY_FULL_CATEGORY_GPU_FIT200_RECONSTRUCTION',ordinary_category_tables_on_GPU=True,
        parameter_bytes=12576064000,all_parameter_dtypes='torch.float32',full_original_backbone_generations=4,
        peak_allocated_bytes=13302261232))
    p.write_text(json.dumps(actual));s['memory_execution_reference']['sha256']=digest(p)
    assert task.validate_memory(s)['proof']['ordinary_category_tables_on_GPU']is True
    # A smaller allocator cannot borrow the larger route's measured capacity.
    with pytest.raises(PermissionError):task.validate_memory(dict(s,allocator_limit_bytes=12*1024**3))
    actual['worker_exit_code']=1;p.write_text(json.dumps(actual));s['memory_execution_reference']['sha256']=digest(p)
    with pytest.raises(PermissionError):task.validate_memory(s)
