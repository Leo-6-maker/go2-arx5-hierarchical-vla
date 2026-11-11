"""Synthetic metadata checks; positive native data requires actual sealed runs."""
import json
import pytest
from test_task_learning_collection_v1 import plan,outcomes,save
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_collection_v1 as collection
from go2arx5_vla.world_model import task_learning_export_v1 as export


def group(root,panel,pin,selection,count=None,finite=False):
    values=outcomes(panel)[selection[0]:selection[1]]
    if count is not None:values=values[:count]
    inp=dict(task_learning_role=task.COLLECTION_ROLE,source_sha=panel['source_sha'],
        panel_sha256=pin,assignment_slice=selection)
    ip=save(root/'INPUT.private.json',inp)
    save(root/'control/LEASE.json',dict(status='CLOSED',input_sha256=ip['sha256'],worker_exit_code=int(finite)))
    result=dict(exit_code=int(finite),source_sha=panel['source_sha'],panel_sha256=pin,
        assignment_slice=selection,selected_requests=selection[1]-selection[0],completed_in_group=len(values),
        summary=collection.summarize(panel,values),
        status='TASK_LEARNING_NATIVE_GROUP_INCOMPLETE_PRESERVED'if finite else'TASK_LEARNING_NATIVE_GROUP_COMPLETE',
        failure='RuntimeError: FINITE_PANEL_NO_LATE_NEW_EPISODE; REMAINING_REQUESTS_NOT_RUN'if finite else None)
    save(root/'RESULT.private.json',result);save(root/'OUTCOMES.private.json',values)
    return values


def test_closed_fifty_case_group_is_not_the_complete_hundred_population(tmp_path):
    p=plan(tmp_path/'proofs',task.COLLECTION_ROLE);pp=save(tmp_path/'PANEL.json',p)
    g=tmp_path/'group';group(g,p,pp['sha256'],[0,50])
    values,_=export.terminal_group(g,p,pp['sha256']);assert len(values)==50
    with pytest.raises(ValueError,match='ALL100_ATTEMPTS'):
        export.build_train_base(tmp_path/'PANEL.json',[g],tmp_path/'data')
    assert not(tmp_path/'data').exists()


def test_explicit_finite_closed_prefix_preserves_remaining_unrun_cases(tmp_path):
    p=plan(tmp_path/'proofs',task.COLLECTION_ROLE);pp=save(tmp_path/'PANEL.json',p)
    g=tmp_path/'group';group(g,p,pp['sha256'],[0,50],count=40,finite=True)
    values,_=export.terminal_group(g,p,pp['sha256']);assert len(values)==40
    assert not collection.summarize(p,values)['complete']


@pytest.mark.parametrize('change',['wrong_order','wrong_role','unknown_stop','not_closed','wrong_summary'])
def test_groups_reject_changed_roles_unknown_stops_or_fabricated_counts(tmp_path,change):
    p=plan(tmp_path/'proofs',task.COLLECTION_ROLE);pp=save(tmp_path/'PANEL.json',p)
    g=tmp_path/'group';group(g,p,pp['sha256'],[0,50],count=40,finite=True)
    if change=='wrong_order':
        path=g/'OUTCOMES.private.json';v=json.loads(path.read_text());v[0],v[1]=v[1],v[0];save(path,v)
    elif change=='wrong_role':
        path=g/'INPUT.private.json';v=json.loads(path.read_text());v['task_learning_role']=task.QUALIFICATION_ROLE;save(path,v)
    elif change=='unknown_stop':
        path=g/'RESULT.private.json';v=json.loads(path.read_text());v['failure']='UNRESOLVED_ENGINEERING_FAILURE';save(path,v)
    elif change=='not_closed':
        path=g/'control/LEASE.json';v=json.loads(path.read_text());v['status']='ACTIVE';save(path,v)
    elif change=='wrong_summary':
        path=g/'RESULT.private.json';v=json.loads(path.read_text());v['summary']['task_successes']=100;save(path,v)
    with pytest.raises(ValueError):export.terminal_group(g,p,pp['sha256'])


def test_same_conditions_cannot_be_recounted_across_two_groups(tmp_path):
    p=plan(tmp_path/'proofs',task.COLLECTION_ROLE);pp=save(tmp_path/'PANEL.json',p)
    g=tmp_path/'group';group(g,p,pp['sha256'],[0,50])
    with pytest.raises(ValueError,match='NO_REPEATED_CASE'):
        export.audit_population(tmp_path/'PANEL.json',[g,g])


def test_synthetic_complete_metadata_without_native_runs_never_creates_training_data(tmp_path):
    p=plan(tmp_path/'proofs',task.COLLECTION_ROLE);pp=save(tmp_path/'PANEL.json',p)
    g=tmp_path/'group';group(g,p,pp['sha256'],[0,100])
    with pytest.raises(FileNotFoundError):
        export.build_train_base(tmp_path/'PANEL.json',[g],tmp_path/'data')
    assert not(tmp_path/'data').exists()
