import copy
import pytest
from go2arx5_vla.world_model import task_learning_expansion_v1 as expansion
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_scene_v1 as scenes
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as teacher


def panel():
    return dict(schema=expansion.SCHEMA, source_sha='a'*40,
        task_learning_role=task.EXPANSION_QUALIFICATION_ROLE,
        protocol=task.protocol(task.EXPANSION_QUALIFICATION_ROLE),
        Teacher_implementation_source_sections_sha256=teacher.FROZEN_TEACHER,
        assignments=[dict(native_run='feasibility-'+layout.lower(), layout=layout,
            parent=scenes.scene_id(layout), split='TRAIN', restart=1,
            command=task.COMMAND, purpose=expansion.PURPOSE,
            Teacher_mode='NORMAL_TEACHER_V7', model_sha256='b'*64)
            for layout in expansion.NATIVE_LAYOUTS])


def test_only_object_position_varies_and_old_panels_are_disjoint():
    base=scenes.geometry()
    old=set(scenes.TRAIN_OFFSETS)|set(scenes.DEV_OFFSETS)|set(scenes.QUALIFICATION_OFFSETS)
    assert not old.intersection(expansion.OFFSETS)
    for layout,offset in expansion.OFFSETS.items():
        geometry=scenes.geometry(layout)
        assert geometry.pop('object_initial_xyz_m')==[.26+offset[0],-.014+offset[1],.62]
        assert geometry.pop('role')=='NEXT_TRAIN_LAYOUT_FEASIBILITY_NOT_BC'
        expected=copy.deepcopy(base);expected.pop('object_initial_xyz_m');expected.pop('role')
        assert geometry==expected
    a,b=task.protocol(),task.protocol(task.EXPANSION_QUALIFICATION_ROLE)
    a.pop('stage');b.pop('stage');assert a==b
    assert teacher.teacher_fingerprint()==teacher.FROZEN_TEACHER


@pytest.mark.parametrize('change', ['DEV','restart','recovery','order','missing'])
def test_population_cannot_be_relabelled_retried_or_selected(change):
    p=panel();row=p['assignments'][0]
    if change=='DEV':row['split']='DEV'
    elif change=='restart':row['restart']=2
    elif change=='recovery':row['Teacher_mode']='UNDER_CLOSE'
    elif change=='order':p['assignments'].reverse()
    else:p['assignments'].pop()
    with pytest.raises(ValueError):expansion.validate(p)


def test_partial_feasibility_does_not_treat_unrun_as_failure_or_admit_bc():
    p=panel();row=p['assignments'][0]
    outcome=dict(row,attempted=True,reset_ready=True,technical_valid=True,
        hard_safety_stop=False,task_success=True)
    result=expansion.summarize(p,[outcome])
    assert result['requested']==6 and result['attempted']==1 and not result['complete']
    assert result['by_absolute_X_offset_cm']['1']['success_per_attempted']==1
    assert result['by_absolute_X_offset_cm']['2']['success_per_attempted'] is None
    assert result['qualified_training_demonstrations']==0
    assert not result['large_collection_authorized'] and not result['reliability_established']
    with pytest.raises(ValueError):expansion.summarize(p,[outcome,outcome])


def test_native_admission_routes_new_role_without_construction_relabelling(monkeypatch):
    from go2arx5_vla.world_model import task_learning_grasp_native_v1 as native
    p=panel();row=p['assignments'][0]
    proof=dict(parent=row['parent'],split='TRAIN')
    monkeypatch.setattr(task,'declared',lambda s:None)
    monkeypatch.setattr(scenes,'admission',lambda s:proof)
    monkeypatch.setattr(native.scenes,'pinned',lambda d:p)
    monkeypatch.setattr(native,'digest',lambda path:'b'*64)
    spec=dict(registered_root='/run/'+row['native_run'],data_role='TRAIN',
        data_parent=row['parent'],scene_id=row['parent'],
        task_learning_role=task.EXPANSION_QUALIFICATION_ROLE,
        task_learning_teacher_mode='NORMAL_TEACHER_V7',model_path='model',
        stable_task_panel='panel',stable_task_panel_sha256='c'*64)
    assert native.scene_admission(spec)==proof
    assert native.native_result_schema(spec)==expansion.NATIVE_SCHEMA
    spec['task_learning_teacher_mode']='UNDER_CLOSE'
    with pytest.raises(ValueError):native.scene_admission(spec)
