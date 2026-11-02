"""CPU protocol/record fixtures; no models, rendering or native admission."""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path

import numpy as np
import pytest

from scripts.pi0294.multitask import shared_evaluation as shared
from scripts.pi0294.multitask.shared_policy import qualified_inference_entry
from scripts.pi0294.multitask.shared_training import recipe
from scripts.pi0294.multitask.raw_parent_delivery import PARENT_ID


def population():
    tasks={t:[]for t in shared.TASKS}
    for task in('T1','T2'):
        for i in range(12):
            for bank in(294101,294102):
                tasks[task].append(dict(id=f'{task}-{i:03}-N{bank}',task=task,split='DEV',
                    noise_bank=bank,noise_namespace=f'PI0294/PV1/DEV/N{bank}',scene_group=f'SCENE/{i:03}',instruction=task))
    from scripts.pi0294.multitask.seed_set_export import DEV_OFFSETS
    tasks['BOOK']=[dict(id=f'BOOK_DEV{i:03}',task='BOOK',split='DEV',xy_offset_m=list(xy))for i,xy in enumerate(DEV_OFFSETS)]
    tasks['DOOR']=[dict(id=f'DOOR_DEV{i:03}',task='DOOR',study_split='DEV',instruction='Open the cabinet door.')for i in range(10)]
    return dict(schema='SHARED4_COMPLETE_ORIGINAL_TASK_DEV_REFERENCE_V2',tasks=tasks,blind_TEST=False,DEV_used_for_statistics_or_optimizer=False)


def sealed():
    pop=population();panel=shared.plan(pop,population_sha256=shared.digest(pop),checkpoint_manifest_sha256='a'*64)
    templates={t:dict(root='/fixture/'+t,manifest_sha256='b'*64)for t in shared.TASKS}
    q=dict(schema='SHARED4_SOURCE_RESET_NATIVE_QUALIFICATION_V1',passed=True,uid=os.getuid(),host=os.uname().nodename,
        numpy_version=np.__version__,MuJoCo_version='3.2.7',checks=dict.fromkeys(shared.NATIVE_CHECKS,True),
        source_files=shared.source_files(),task_templates=templates)
    m=dict(kind=shared.KIND,limits=deepcopy(shared.LIMITS),condition_index=0,run_uuid=shared.condition_run_name(panel,0),
        raw_zero_step=False,evaluation_checkpoint_manifest_sha256='a'*64,task_templates=templates,
        sealed_inputs={'SHARED_DEV_PLAN.json':shared.digest(panel),'TASK_DEV_POPULATION.json':shared.digest(pop),
            'TASK_NATIVE_QUALIFICATION.json':shared.digest(q),'RESET_PREDECESSOR.json':'c'*64})
    return m,panel,pop,q


def test_all68_original_conditions_two_noise_banks_and_no_subset_or_retry():
    m,p,pop,q=sealed();assert shared.validate_manifest(m,p,pop,q)==pop['tasks']['T1'][0]
    assert len(p['conditions'])==68 and p['task_counts']==dict(T1=24,T2=24,BOOK=10,DOOR=10)
    assert len({shared.condition_run_name(p,i)for i in range(68)})==68
    assert p['one_registered_process_per_condition']and not p['same_condition_retry']
    for fault in('missing','TRAIN','noise','duplicate'):
        bad=deepcopy(pop)
        if fault=='missing':bad['tasks']['DOOR'].pop()
        elif fault=='TRAIN':bad['tasks']['BOOK'][0]['split']='TRAIN'
        elif fault=='noise':bad['tasks']['T1'][0]['noise_bank']=999
        else:bad['tasks']['T2'][1]['id']=bad['tasks']['T2'][0]['id']
        with pytest.raises(PermissionError):shared.plan(bad,population_sha256=shared.digest(bad),checkpoint_manifest_sha256='a'*64)


@pytest.mark.parametrize('fault',['model','index','retry','qualification','source','numpy','template','predecessor','extra_training','raw_alias'])
def test_changed_model_process_or_unqualified_native_reset_never_admits(fault):
    m,p,pop,q=sealed()
    if fault=='model':m['evaluation_checkpoint_manifest_sha256']='d'*64
    elif fault=='index':m['condition_index']=True
    elif fault=='retry':m['run_uuid']+='-retry'
    elif fault=='qualification':q['checks']['live_predecessor_rejected']=False
    elif fault=='source':q['source_files'][shared.SOURCES[0]]='d'*64
    elif fault=='numpy':q['numpy_version']='other'
    elif fault=='template':q['task_templates']=dict(q['task_templates'],BOOK={})
    elif fault=='predecessor':del m['sealed_inputs']['RESET_PREDECESSOR.json']
    elif fault=='extra_training':m['limits']['optimizer_updates']=1
    else:m['raw_zero_step']=True
    m['sealed_inputs']['TASK_NATIVE_QUALIFICATION.json']=shared.digest(q)
    with pytest.raises(PermissionError):shared.validate_manifest(m,p,pop,q)


def test_model_independent_original_noise_and_student_projection():
    pop=population()
    for task in shared.TASKS:
        c=pop['tasks'][task][0];s=shared.SharedStudent(None,c,None,None)
        key=c['scene_group']+'/'+task if task in('T1','T2')else c['id']
        for i in(0,1,79):
            expected=int.from_bytes(hashlib.sha256(f'{s.seed_namespace}/{key}/{i}'.encode()).digest()[:8],'big')%(2**63)
            assert s.generation_seed(i)==expected
    captures=[]
    class Model:
        def generate(self,request,seed):
            captures.append((request,seed));values=np.tile(request['state_eef10'],(40,1))
            return dict(physical_H40=values)
    c=pop['tasks']['T1'][0];student=shared.SharedStudent(Model(),c,lambda *a:None,lambda:None)
    pose=np.eye(4);pose[:3,3]=[.4,-.1,.55]
    student.observe(dict(tcp=pose,gripper_fraction=1.,time_s=5.,object_pose='PRIVILEGED_NOT_AN_INPUT',teacher_stage='PRIVATE'))
    student.set_images({n:np.zeros((256,256,3),dtype=np.uint8)for n in('body_rgb','wrist_rgb')},c['instruction'])
    student.plan();request,seed=captures[0]
    assert set(request)=={'snapshot_uuid','state_eef10','images','language','frame','tcp'}
    assert 'PRIVILEGED'not in json.dumps(request)and request['language']==c['instruction']
    with pytest.raises(PermissionError):student.plan()


def test_macro_and_worst_task_are_not_invented_from_truncated_panels():
    _,p,_,_=sealed();rows=[]
    for c in p['conditions']:
        rows.append(dict(task=c['task'],id=c['id'],valid_model_condition=True,success=True,hard_native_safety=False))
    assert shared.summarize(p,rows)['macro_success_rate']==1.
    rows[0]['success']=False;rows[0]['hard_native_safety']=True
    summary=shared.summarize(p,rows);assert summary['tasks']['T1']['hard_safety_stops']==1
    assert summary['worst_task_success_rate']==23/24
    partial=shared.summarize(p,rows[:1]);assert partial['macro_success_rate']is None and not partial['complete_all68']
    assert partial['tasks']['DOOR']['unrun']==10
    with pytest.raises(PermissionError):shared.summarize(p,rows+rows[:1])


def inference_manifest():
    p=recipe(*['1'*64,'2'*64,'3'*64,'4'*64])
    return dict(kind=shared.INFERENCE_KIND,limits=shared.INFERENCE_LIMITS,raw_parent_files_identity=PARENT_ID,
        raw_zero_step=True,evaluation_checkpoint_manifest_sha256=PARENT_ID,GPU_uuid='GPU-ONE',initialization_recipe=p)


def test_raw_and_trained_inference_require_own_closed_same_device_native16(tmp_path):
    q=inference_manifest();assert qualified_inference_entry(q)==13*1024**3
    documents={'RUN_MANIFEST.json':q,'RESULT.json':dict(status='COMPLETE_SHARED_TRAIN16_INFERENCE_QUALIFICATION',error=None,
        model_load_attempts=1,optimizer_updates=0,physics_integrations=0,generation_attempts=16,completed_generations=16,formal_DEV_conditions=0),
        'SHARED_GPU_INFERENCE_CAPACITY.json':dict(capacity_pass=True,minimum_free_bytes=1024**3,measurements=17),
        'ACTUAL_SHARED_INFERENCE_RECONSTRUCTION.json':dict(evaluation_checkpoint_manifest_sha256=PARENT_ID,frozen_parent_unchanged=True)}
    documents['control/LEASE.json']=dict(status='CLOSED',manifest_sha256=shared.digest(q),result_sha256=shared.digest(documents['RESULT.json']))
    for n,v in documents.items():
        f=tmp_path/n;f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(v,indent=2)+'\n')
    m=dict(q,kind=shared.KIND,limits=shared.LIMITS,shared_inference_admission=dict(schema='SHARED4_SAME_GPU_NATIVE16_INFERENCE_V1',
        qualification_root=str(tmp_path),files={n:shared.digest(v)for n,v in documents.items()}))
    assert qualified_inference_entry(m)==13*1024**3
    for field,value in [('GPU_uuid','GPU-TWO'),('evaluation_checkpoint_manifest_sha256','a'*64),('raw_zero_step',False),
                        ('GPU_training_admission',{}),('parent_checkpoint','BOOK128')]:
        bad=deepcopy(m);bad[field]=value
        with pytest.raises(PermissionError):qualified_inference_entry(bad)
    documents['SHARED_GPU_INFERENCE_CAPACITY.json']['minimum_free_bytes']-=1
    f=tmp_path/'SHARED_GPU_INFERENCE_CAPACITY.json';f.write_text(json.dumps(documents[f.name],indent=2)+'\n')
    m['shared_inference_admission']['files'][f.name]=shared.digest(documents[f.name])
    with pytest.raises(PermissionError):qualified_inference_entry(m)


def test_candidate_preflight_rejects_book_native10_and_incomplete_producer_before_model(tmp_path,monkeypatch):
    from scripts.pi0294.multitask.shared_policy import candidate_custody
    from scripts.pi0294.multitask import book_reset,shared_checkpoint
    from scripts.pi0294.e1 import smoke
    monkeypatch.setattr(smoke,'node_local',Path)
    monkeypatch.setattr(book_reset,'verify_predecessor',lambda *a,**kw:None)
    m=inference_manifest();assert candidate_custody(tmp_path,m)is None
    m['evaluation_checkpoint_manifest_sha256']='a'*64
    with pytest.raises(PermissionError):candidate_custody(tmp_path,m)
    producer=tmp_path/'producer';producer.mkdir();m.update(raw_zero_step=False,delta_producer=dict(root=str(producer)))
    for status,steps in [('COMPLETE_SHARED_RAW_FINITE_LEARNING',10),('COMPLETE_BOOK5_FINITE_TARGET_LEARNING',8192),
                         ('FAILED_SHARED_RAW_FINITE_LEARNING',8192)]:
        (producer/'RESULT.json').write_text(json.dumps(dict(status=status,optimizer_updates=steps,error=None)))
        with pytest.raises(PermissionError,match='COMPLETE_CLOSED_SHARED_PRODUCER'):candidate_custody(tmp_path,m)


def test_invalid_initial_gripper_is_rejected_before_any_files_or_constructor(tmp_path):
    from scripts.pi0294.multitask.book_policy import episode
    for value in(float('nan'),True,-.01,1.01):
        with pytest.raises(PermissionError,match='INITIAL_GRIPPER'):
            episode(tmp_path,{},None,None,None,warmup_opening=value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('task',['T1','T2'])
@pytest.mark.parametrize('passed',[True,False])
def test_original_cube_target_visibility_uses_the_existing_qualification_pair(monkeypatch,task,passed):
    from types import SimpleNamespace
    from scripts.pi0294 import scene
    images,calibration={},{};seen=[]
    def original(a,b,card):
        assert a is images and b is calibration and card['task']==task
        seen.append(True);return dict(passed=passed)
    monkeypatch.setattr(scene,'rendered_field_of_view',original)
    engine=SimpleNamespace(card={'task':task})  # No render method: a second render is forbidden.
    if passed:assert shared.qualification_extra(engine,None,images,calibration)['original_cube_camera_qualification']['passed']
    else:
        with pytest.raises(PermissionError,match='FULL_TARGET_VIEW'):shared.qualification_extra(engine,None,images,calibration)
    assert len(seen)==1


def cube_rows(task):
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS
    def state(t):
        grasp=bool(task=='T1'and 1<=t<=4);obj=np.eye(4);tcp=np.eye(4)
        obj[:3,3]=[-.16 if t<2.5 else .16,0.,.54 if task=='T1'and 1.5<=t<=4 else .46]
        tcp[:3,3]=[.16,0.,.65]
        s={n:False for n in BOOL_FIELDS}
        s.update({n:True for n in('object_present','contact_valid','gripper_state_valid','safety_valid')})
        s.update({n:grasp for n in('native_grasp','left_object_contact','right_object_contact')})
        return dict(s,time_s=5+t,object_pose=obj.tolist(),tcp=tcp.tolist(),object_linear_velocity=[0.,0.,0.],
            object_angular_velocity=[0.,0.,0.],robot_object_contact=bool(1<=t<=4))
    rows=[dict(kind='observation',value=dict(time_s=5+i/20,actual_state=state(i/20)))for i in range(641)]
    rows += [dict(kind='native_tick',value=dict(actual_state=state(i/500)))for i in range(1,16001)]
    return rows,dict(task=task,task_id=task,goal_xy=[.16,0.])


@pytest.mark.parametrize('task',['T1','T2'])
def test_saved_cube_projection_preserves_original_complete_task_predicates(task):
    from scripts.pi0294.multitask.cube_raw_score import score
    rows,card=cube_rows(task);result=score(rows,card,complete=True)
    assert result['persistent_success']and result['hold_pass']==41
    assert not score(rows,card,complete=False)['persistent_success']
    if task=='T2':
        for row in rows:
            s=row['value']['actual_state']
            if 7<=s['time_s']<=7.5:
                for n in('native_grasp','left_object_contact','right_object_contact'):s[n]=True
        result=score(rows,card,complete=True)
        assert not result['persistent_success']and not result['history']['no_sustained_bilateral_grasp']
