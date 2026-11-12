import copy
import hashlib
import json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import native_a500_diagnostic_v1 as d
from tests.world_model.test_native_arx5_diagnostic_v1 import panel as old_panel


def panel():
    p=old_panel();p.update(schema=d.SCHEMA,task_learning_role=d.PROTOCOL,protocol=d.protocol(),
        B1_global_step=500,preregistration=dict(sha256=d.CONTRACT_SHA),policies={d.METHODS[0]:dict(arm=d.METHODS[0])})
    for r in p['assignments']:r.update(method=d.METHODS[0],purpose=d.PURPOSE)
    return p


def test_a500_has_own_fixed_four_and_preregistered_score_without_changing_old_role():
    p=panel();assert task.native_body_role(d.PROTOCOL)and task.student_role(d.PROTOCOL)
    assert task.protocol(d.PROTOCOL)==d.protocol()
    assert len(d.validate_panel(p))==4
    f=Path('experiments/native_arx5_transfer/a_target500_train4_delivery_v1.json')
    assert hashlib.sha256(f.read_bytes()).hexdigest()==d.CONTRACT_SHA
    actual=json.loads(f.read_bytes());assert actual['score_contract']==task.score_contract()
    assert actual['maximum_queries']==60 and actual['original_DEV_TEST_conditions']==0
    from go2arx5_vla.world_model import native_arx5_diagnostic_v1 as old
    assert old.validate_panel(old_panel())and old.PROTOCOL!=d.PROTOCOL
    for key,value in [('B1_global_step',250),('B1_global_step',2000),('schema',old.SCHEMA)]:
        with pytest.raises(ValueError):d.validate_panel(dict(p,**{key:value}))


def test_a500_diagnostic_never_accepts_dev_or_selected_subset_or_old_policy():
    p=panel();inp=dict(schema=d.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,4],population_predecessor=None)
    assert len(d.selection(inp,p))==4
    with pytest.raises(ValueError):d.selection(dict(inp,assignment_slice=[0,1]),p)
    wrong=copy.deepcopy(p);wrong['assignments'][0]['split']='DEV'
    with pytest.raises(ValueError):d.validate_panel(wrong)
    with pytest.raises(ValueError):d.policy_contract(dict(arm='SFT2000_PLUS_ENDPOINT_BASIS',package={}),{})


def test_pair_spec_routes_official_parent_and_own_saved_target_package(tmp_path):
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import specs
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    p=panel();row=p['assignments'][0]
    scene=pin('scene.json',dict(layout=row['layout'],parent=row['parent'],compiled=dict(scene_sha256=row['model_sha256'])))
    audit=pin('audit.json',dict(official_parent='/official/GR00T-N1.7-3B'))
    rgb=dict(path='/own/A500/RECONSTRUCTION_ACTUAL.private.json',sha256='4'*64)
    admission=pin('admission.json',dict(allocator_limit_bytes=13*1024**3,official_parent_audit=audit,
        ordinary_parameter_bytes=12576064000,native_camera_dataset='/body-wrist-data',ordinary_RGB_reconstruction=rgb))
    p['policies'][d.METHODS[0]]=dict(arm=d.METHODS[0],Native_body_qualification=admission)
    inp=dict(source_root=str(tmp_path/'source'),source_sha='2'*40,source_files={},scenes={row['layout']:scene},
        pilot1_binding='/binding',pilot1_binding_sha256='3'*64,panel='/panel',panel_sha256='5'*64,
        scene_allowlist='/scenes',scene_allowlist_sha256='6'*64,fusion_binding='/fusion',fusion_binding_sha256='7'*64,
        memory_execution='LEGACY_TEMPLATE_ONLY',memory_execution_reference={})
    n,g=specs(inp,p,row,dict(native_environment={}),dict(neural_bindings=dict(source='/upstream',dataset='/old')),
              tmp_path/'native',tmp_path/'gpu',2000000000.)
    assert n['task_learning_role']==g['core_local_protocol']==d.PROTOCOL
    assert n['data_role']=='TRAIN'and n['target_gr00t_B1_global_step']==500
    assert g['checkpoint']=='/official/GR00T-N1.7-3B'and g['checkpoint_files']==d.official_files()
    assert g['checkpoint_manifest_sha256']==d.PARENT_ID
    assert g['neural_bindings']['dataset']=='/body-wrist-data'
    assert g['memory_execution_reference']==rgb
