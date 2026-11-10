import copy
import pytest
from go2arx5_vla.world_model import task_learning_conic_native_v3 as c
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id


def panel():
    rows=[]
    for i,(layout,method,seed)in enumerate(c.ORDER):
        rows.append(dict(native_run=f'new-{i}',layout=layout,method=method,parent=scene_id(layout),split='TRAIN',
            purpose=c.PURPOSE,restart=1,command=task.COMMAND,query_seeds=list(range(seed,seed+15)),model_sha256='a'*64))
    return dict(schema=c.SCHEMA,task_learning_role=c.PROTOCOL,protocol=task.protocol(c.PROTOCOL),B1_global_step=2000,
        preregistration={'sha256':c.CONTRACT_SHA},source_sha='b'*40,policies={m:{'arm':m}for m in c.METHODS},assignments=rows)


def test_fixed_population_full_denominators_and_old_namespace_rejects():
    p=panel();inp=dict(schema=c.INPUT_SCHEMA,assignment_slice=[0,4],population_predecessor=None,source_sha=p['source_sha'])
    assert batch.selection(inp,p)==p['assignments']
    report=c.summarize(p,[])
    assert report['requested']==4 and report['completed']==0 and not report['complete']
    assert all(v['requested']==2 and v['attempted']==0 for v in report['methods'].values())
    from go2arx5_vla.world_model.task_learning_local_policy_v1 import validate_panel as old_local
    from go2arx5_vla.world_model.task_learning_student_v1 import validate as dev
    for validate in(old_local,dev):
        with pytest.raises(ValueError):validate(p)
    for key,new in [('layout','TRAIN01'),('method','B1'),('query_seeds',list(range(15)))]:
        bad=copy.deepcopy(p);bad['assignments'][0][key]=new
        with pytest.raises(ValueError):c.validate_panel(bad)


def test_no_duplicate_outcome_or_partial_retry():
    p=panel();r=dict(p['assignments'][0],attempted=True,task_success=False,technical_valid=True,hard_safety_stop=False,
        actual_GR00T_queries=1,actual_native_integrations=5703)
    report=c.summarize(p,[r]);assert report['methods']['C500_RAW']['valid_task_failures']==1
    with pytest.raises(ValueError):c.summarize(p,[r,r])
    with pytest.raises(ValueError):c.selection(dict(schema=c.INPUT_SCHEMA,assignment_slice=[1,4],population_predecessor=None,source_sha=p['source_sha']),p)


def test_new_role_is_student_without_teacher_protocol_changes():
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert task.student_role(task.CONIC_ROLE)
    assert task.protocol(task.CONIC_ROLE)['expert_commands']=='NONE_STUDENT_NOT_EXPERT'
    assert task.protocol(task.CONIC_ROLE)['physics_dt_s']==.002
    assert teacher_fingerprint()==FROZEN_TEACHER
