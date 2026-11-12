import copy
import pytest
from go2arx5_vla.world_model import native_arx5_diagnostic_v1 as d
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id


def panel():
    rows=[dict(layout=layout,method=d.METHODS[0],parent=scene_id(layout),restart=1,
               native_run='native-body-'+layout.lower(),split='TRAIN',purpose=d.PURPOSE,
               command=task.COMMAND,query_seeds=list(range(seed,seed+15)),model_sha256='1'*64)
          for layout,seed in d.ORDER]
    return dict(schema=d.SCHEMA,task_learning_role=d.PROTOCOL,protocol=d.protocol(),
                source_sha='2'*40,B1_global_step=2000,preregistration=dict(sha256=d.CONTRACT_SHA),
                policies={d.METHODS[0]:dict(arm=d.METHODS[0])},assignments=rows)


def test_native_body_diagnostic_is_exact_four_train_conditions():
    p=panel()
    assert task.student_role(d.PROTOCOL)
    assert task.protocol(d.PROTOCOL)==d.protocol()
    assert [r['layout'] for r in d.validate_panel(p)]==['TRAIN01','TRAIN07','TRAIN13','TRAIN25']
    inp=dict(schema=d.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,4],population_predecessor=None)
    assert len(d.selection(inp,p))==4
    for span in ([0,2],[1,4],[0,80]):
        with pytest.raises(ValueError):d.selection(dict(inp,assignment_slice=span),p)
    for key,value in [('schema','CORE_MATCHED128_SMOOTH_NATIVE_PANEL_V3'),('B1_global_step',500)]:
        with pytest.raises(ValueError):d.validate_panel(dict(p,**{key:value}))
    bad=copy.deepcopy(p);bad['assignments'][0]['split']='DEV'
    with pytest.raises(ValueError):d.validate_panel(bad)


def test_old_admission_cannot_be_used_as_new_body_policy():
    with pytest.raises(ValueError,match='BODY_FROZEN_SFT2000'):
        d.policy_contract(dict(arm='B1',parent_manifest_sha256='1'*64,package=None,Core_local_qualification=None),{})


def test_summary_retains_rejection_and_technical_failure_denominators():
    p=panel();out=[]
    for i,row in enumerate(p['assignments']):
        out.append(dict(row,attempted=True,technical_valid=i<3,task_success=i==0,hard_safety_stop=False,
                        actual_GR00T_queries=i+1,actual_native_integrations=5703))
    result=d.summarize(p,out)
    assert result['complete']
    m=result['methods'][d.METHODS[0]]
    assert (m['requested'],m['attempted'],m['task_successes'],m['valid_task_failures'],m['technical_failures'])==(4,4,1,2,1)
    assert result['original_DEV_conditions']==result['TEST_conditions']==result['model_updates']==0
    with pytest.raises(ValueError):d.summarize(p,[out[1],out[0]])
