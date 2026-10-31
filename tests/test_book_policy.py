from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.multitask.book_policy import Student,ModelReferenceRejected,dev_card
from scripts.pi0294.multitask.seed_set_export import DEV_OFFSETS


def test_dev_changes_initial_book_only_and_preserves_full_original_task():
    cards=[dev_card(dict(id=f'BOOK_DEV{i:03}',xy_offset_m=xy))for i,xy in enumerate(DEV_OFFSETS)]
    home=cards[0]['initial_tcp'];goal=cards[0]['book']['target_book_world']
    for card,xy in zip(cards,DEV_OFFSETS):
        assert card['initial_tcp']==home and card['book']['target_book_world']==goal
        assert card['warmup_s']==5 and card['task_duration_s']==32 and card['study_split']=='DEV'
        np.testing.assert_allclose(np.array(card['initial_book_native_world'])[:2,3],[.36+xy[0],-.10+xy[1]])
    with pytest.raises(ValueError):dev_card(dict(id='BOOK_DEV010',xy_offset_m=[0,0]))


def test_student_projects_only_proprio_rgb_language_and_rejects_without_command_modification():
    observed=[];records=[];state=dict(tcp=np.eye(4),gripper_fraction=.9,time_s=5,
        object_pose='PRIVILEGED_OBJECT_POSE',native_grasp='PRIVILEGED_CONTACT',teacher_stage='PRIVILEGED_STAGE')
    image=np.zeros((256,256,3),np.uint8)
    class Model:
        def generate(self,request,seed):
            observed.append(deepcopy(request));a=np.tile([0,0,0,1,0,0,0,1,0,.9],(40,1));a[:,0]=np.arange(40)*.2
            return dict(physical_H40=a)
    student=Student(Model(),'BOOK_DEV000',lambda kind,row:records.append((kind,row)),lambda:None)
    student.observe(state);student.set_images(dict(body_rgb=image,wrist_rgb=image),'Put book away')
    with pytest.raises(ModelReferenceRejected):student.plan()
    assert student.queries==1 and len(observed)==1 and len(records)==2
    serialized=str(observed[0]);assert 'PRIVILEGED'not in serialized
    assert set(observed[0])=={'snapshot_uuid','state_eef10','images','language','frame','tcp'}
    assert records[-1][1]['value']['physical_H40'][1,0]==.2
    with pytest.raises(PermissionError):student.plan()
    assert student.queries==1
