"""File/condition-join fixtures, not native execution qualification."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_student_v1 as student
from go2arx5_vla.world_model import task_learning_student_population_v2 as population
from go2arx5_vla.world_model.task_learning_endpoint_decision_v2 import descriptor
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_task_learning_student_v1 import panel


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp_path,*,controlled_rejection=False,technical=False):
    p=panel();pp=write(tmp_path/'panel.json',p);reference=descriptor(pp)
    source=tmp_path/'source';name='src/go2arx5_vla/world_model/task_learning_student_v1.py'
    file=source/name;file.parent.mkdir(parents=True);file.write_bytes(Path(student.__file__).read_bytes())
    groups=[]
    for start in range(0,80,10):
        root=tmp_path/f'group{start:02d}';groups.append(root);values=[]
        inp=dict(schema='TASK_LEARNING_STUDENT_DEV_GROUP_INPUT_V1',source_sha=p['source_sha'],
            panel=str(pp),panel_sha256=reference['sha256'],source_root=str(source),
            source_files={name:digest(file)},assignment_slice=[start,start+10])
        ip=write(root/'INPUT.private.json',inp)
        for row in p['assignments'][start:start+10]:
            n=root/'runs'/row['native_run'];gpu=root/'runs'/(row['native_run']+'-gpu')
            sp=write(n/'EXECUTION_SPEC.json',dict(registered_root=str(n),source_sha=p['source_sha'],
                stable_task_panel_sha256=reference['sha256'],target_gr00t_gpu_root=str(gpu)))
            rejected=controlled_rejection and not values and start==0
            failed=technical and not values and start==0
            code=1 if rejected or failed else 0
            write(n/'control/LEASE.json',dict(status='CLOSED',spec_sha256=digest(sp),worker_exit_code=code))
            write(n/'WORKER_EXIT.private.json',dict(exit_code=code))
            nr=write(n/'native-01/RESULT.json',dict(fixture=True,never_executed=True))
            gs=write(gpu/'EXECUTION_SPEC.json',dict(producer_native_spec=descriptor(sp)))
            write(gpu/'control/LEASE.json',dict(status='CLOSED',spec_sha256=digest(gs),worker_exit_code=0))
            write(gpu/'WORKER_EXIT.private.json',dict(exit_code=0))
            gr=None if failed else write(gpu/'fit-01/RESULT.private.json',dict(fixture=True,never_executed=True))
            values.append(dict(row,attempted=True,reset_ready=True,task_success=False,technical_valid=not failed,
                hard_safety_stop=False,worker_exit_code=code,result_sha256=digest(nr),GPU_result_sha256=digest(gr)if gr else None))
        write(root/'OUTCOMES.private.json',values)
        write(root/'control/LEASE.json',dict(status='CLOSED',input_sha256=digest(ip),worker_exit_code=0))
        write(root/'RESULT.private.json',dict(exit_code=0,source_sha=p['source_sha'],panel_sha256=reference['sha256'],
            assignment_slice=inp['assignment_slice'],selected_requests=10,completed_in_group=10,
            failure=None,summary=student.summarize(p,values)))
    return reference,groups


def test_all_eight_groups_retain_complete_80_but_do_not_issue_training_permission(tmp_path):
    ref,groups=fixture(tmp_path)
    result=population.aggregate(ref,groups)
    assert result['status']=='COMPLETE_COMMON_DEV80'and result['attempted_conditions']==80
    assert result['summary']['methods']['B0']['attempted']==40
    assert result['summary']['methods']['B1']['attempted']==40
    assert result['scientific_continuation_authorized']is False


def test_partial_counts_unrun_and_duplicate_attempted_condition_is_rejected(tmp_path):
    ref,groups=fixture(tmp_path)
    result=population.aggregate(ref,groups[:1])
    assert result['status'].startswith('PARTIAL')and result['unrun_conditions']==70
    with pytest.raises(ValueError,match='COUNTED_TWICE'):population.aggregate(ref,[groups[0],groups[0]])


def test_valid_native_rejection_and_known_missing_GPU_result_stay_negative(tmp_path):
    ref,groups=fixture(tmp_path,controlled_rejection=True)
    assert population.aggregate(ref,groups)['native']['methods']['B0']['technical_failure_conditions']==0
    other=tmp_path/'technical';ref,groups=fixture(other,technical=True)
    assert population.aggregate(ref,groups)['native']['methods']['B0']['technical_failure_conditions']==1


def test_changed_child_evidence_cannot_hide_behind_cached_summary(tmp_path):
    ref,groups=fixture(tmp_path)
    p=next((groups[0]/'runs').glob('*/native-01/RESULT.json'));p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='RESULT_STILL_SEALED'):population.aggregate(ref,groups)
