from copy import deepcopy
import numpy as np
import pytest

from scripts.pi0294.multitask import book_early_recovery as r
from scripts.pi0294.multitask.libero_book_teacher import route_card,seed_card,RELEASE_REVISION


def test_recovery_population_is_only_original_train_and_preserves_scene():
    assert len(r.conditions())==10
    assert sum(c['takeover_index']//8 for c in r.conditions())==15
    for c in r.conditions():
        expected=route_card(RELEASE_REVISION)if c['TRAIN_parent']==0 else seed_card(c['TRAIN_parent'])
        actual=r.recovery_card(c)
        for name in('initial_tcp','initial_book_native_world','book','book_route','instruction','warmup_s','task_duration_s'):
            assert actual[name]==expected[name]
        assert actual['study_split']=='TRAIN_RECOVERY_DIAGNOSTIC'
    with pytest.raises(ValueError):r.recovery_card(dict(id='BOOK_DEV000',xy_offset_m=[.006,0]))


@pytest.mark.parametrize('condition',r.conditions())
def test_real_observed_handover_queries_stop_and_ineligible_prefix_excluded(condition):
    records=[];experts=[]
    class Base:
        def __init__(self,instance,identifier,record,guard,*,seed_namespace):
            assert seed_namespace=='BOOK_M32_EARLY_TRAIN_RECOVERY'
            self.index=-1;self.queries=self.completed_generations=0;self.guard=guard
        def observe(self,s):self.index+=1;self.state=np.asarray(s['state10'])
        def plan(self):self.queries+=1;self.completed_generations+=1;return 'STUDENT'
    class Expert:
        stage='approach';candidate_complete=False
        def __init__(self,card):self.card=card;self.observed=[];experts.append(self)
        def observe(self,s):self.observed.append(deepcopy(s))
        def plan(self):return 'EXPERT'
    student=r.make_student(Base,None,condition['id'],lambda *v:records.append(v),lambda:None,teacher_type=Expert)
    plans=[];states=[];takeover=condition['takeover_index']
    for i in range(takeover+9):
        s=dict(state10=[i]*10,safety_valid=True,native_grasp=False,time_s=5+i*.05)
        states.append(s);student.observe(s)
        if i%8==0:plans.append(student.plan())
    assert plans==['STUDENT']*(takeover//8)+['EXPERT']*2
    assert student.completed_generations==takeover//8
    assert experts[0].observed[0]==states[takeover]
    assert experts[0].card==r.original_card(condition)
    assert records[0][1]['actual_state_eef10']==states[takeover]['state10']
    assert records[0][1]['student_prefix_is_expert']is False
    anchors=r.eligible_recovery_anchors(condition,range(640),full_task_success=True,native_safe=True)
    assert anchors==tuple(range(takeover,601))
    assert r.eligible_recovery_anchors(condition,range(640),full_task_success=False,native_safe=True)==()


def test_early_recovery_manifest_keeps_frozen_checkpoint_and_query_bounds():
    m=dict(kind=r.KIND,recovery_plan=r.plan(),delta_checkpoint_manifest_sha256=r.CHECKPOINT,
           limits=dict(optimizer_updates=0,physics_integrations=185000,generation_queries=15,
                       RGB_pairs=6420,source_constructors=10,model_loads=1,output_bytes=r.OWNER_OUTPUT_BYTES))
    r.validate_manifest(m)
    for key in('physics_integrations','generation_queries','source_constructors'):
        bad=deepcopy(m);bad['limits'][key]+=1
        with pytest.raises(PermissionError):r.validate_manifest(bad)
    bad=deepcopy(m);bad['recovery_plan']['conditions'][0]['id']='BOOK_DEV000'
    with pytest.raises(PermissionError):r.validate_manifest(bad)
