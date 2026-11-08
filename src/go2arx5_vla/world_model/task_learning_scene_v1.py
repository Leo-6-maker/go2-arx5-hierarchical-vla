"""Separate near-workspace curriculum, with the original robot and native safety.

Fresh XML compilation rebuilds static collision geometry. Saved-pose checks are
hypotheses, never reset observations, new transitions or dynamic qualification.
"""
import json
import struct
from pathlib import Path
import numpy as np
from .storage import digest
from .pilot1_curriculum import scene_xml,robot_contract

BASE_SCENE='8fc03629c29a4dda8403fb220b60472670a912013d05d717d1568ace34a4241c'
MEMBER='src/go2arx5_vla/world_model/task_learning_scene_v1.py'
SCENE_ID='TASK_LEARNING_UNDER_HAND_TRAIN_PILOT01'
QUALIFICATION_OFFSETS={
    'QUAL01':(-.004,-.004),'QUAL02':(-.004,0.),'QUAL03':(-.004,.004),
    'QUAL04':(0.,-.004),'QUAL05':(0.,.004),'QUAL06':(.004,-.004),
    'QUAL07':(.004,0.),'QUAL08':(.004,.004),'QUAL09':(-.002,.002),'QUAL10':(.002,-.002)}
TRAIN_OFFSETS={f'TRAIN{i+1:02d}':(x,y)for i,(x,y)in enumerate(
    (x,y)for x in(-.0035,-.00175,.00025,.002,.0035)for y in(-.0035,-.00175,.00025,.002,.0035))}
RECOVERY_OFFSETS={'RECOVERY01':(-.003,-.001),'RECOVERY02':(-.003,.001),
    'RECOVERY03':(.003,-.001),'RECOVERY04':(.003,.001)}


def scene_id(layout):
    if layout=='PILOT01':return SCENE_ID
    if layout not in {**QUALIFICATION_OFFSETS,**TRAIN_OFFSETS,**RECOVERY_OFFSETS}:
        raise ValueError('PREREGISTERED_UNDER_HAND_LAYOUT_ONLY')
    return 'TASK_LEARNING_UNDER_HAND_'+layout


def geometry(layout='PILOT01'):
    scene_id(layout)
    dx,dy=(0.,0.)if layout=='PILOT01'else {**QUALIFICATION_OFFSETS,**TRAIN_OFFSETS,**RECOVERY_OFFSETS}[layout]
    return dict(table_center_xyz_m=[.40,0.,.575],table_half_size_m=[.17,.24,.025],
        object_initial_xyz_m=[.26+dx,-.014+dy,.62],cube_edge_m=.04,cube_mass_kg=.05,
        role=('CONSTRUCTION_ONLY_NOT_TRAINING_OR_EVALUATION_POPULATION'if layout=='PILOT01'else
              'CONTINUOUS_TEACHER_TRAIN_POPULATION'if layout in TRAIN_OFFSETS else
              'UNDER_CLOSE_RECOVERY_QUALIFICATION_NOT_BC'if layout in RECOVERY_OFFSETS else'TEACHER_QUALIFICATION_NOT_BC'),
        robot_startup_native_safety_or_dynamics_changed=False)


def compile_and_check(spec,output):
    import mujoco
    from scripts.pi0294.task_state import ActualState,parse_state
    from scripts.pi0294.mp_client import contact_permitted
    from .native_replay import forbid_integration
    output=Path(output)
    layout=spec.get('layout','PILOT01')
    if (mujoco.__version__!='3.2.7' or digest(spec['original_model'])!=BASE_SCENE
        or spec['geometry']!=geometry(layout)or output.exists()):raise ValueError('EXACT_FRESH_NEAR_WORKSPACE_STATIC_INPUTS')
    for name,pin in spec['asset_input_pins'].items():
        if digest(Path(spec['asset_root'])/name)!=pin:raise ValueError('FROZEN_NATIVE_XML_ASSET')
    if digest(spec['saved_startup_journal'])!=spec['saved_startup_journal_sha256']:
        raise ValueError('PINNED_CLOSED_SAVED_STARTUP_JOURNAL')
    original=mujoco.MjModel.from_binary_path(spec['original_model'])
    text=scene_xml(dict(geometry(layout),asset_root=spec['asset_root']),original)
    model=mujoco.MjModel.from_xml_string(text)
    contract=robot_contract(original,model)
    offset=model.key_qpos[0,27:30]-original.key_qpos[0,27:30]
    output.mkdir()
    xml=output/'SCENE.private.xml';xml.write_text(text)
    path=output/'SCENE.private.mjb';mujoco.mj_saveModel(model,str(path),None)
    model=mujoco.MjModel.from_binary_path(str(path));robot_contract(original,model)
    reader=ActualState(model);count=0;first=None
    # This is the real original startup journal in a separate FK reader, with
    # only hypothetical object placement. No robot value or source is edited.
    with forbid_integration(mujoco)as integrations,Path(spec['saved_startup_journal']).open('rb')as stream:
        while header:=stream.read(4):
            if len(header)!=4:raise ValueError('COMPLETE_SAVED_STARTUP_HEADER')
            length,=struct.unpack('<I',header)
            if not 40<=length<=2*1024**2:raise ValueError('BOUNDED_ORIGINAL_PACKET')
            raw=stream.read(length)
            if len(raw)!=length:raise ValueError('COMPLETE_SAVED_STARTUP_PACKET')
            packet=parse_state(raw);count+=1
            if packet['step']!=count:raise ValueError('CONTIGUOUS_ORIGINAL_STARTUP')
            if count>5703:break
            hypothetical=dict(packet,qpos=packet['qpos'].copy())
            hypothetical['qpos'][27:30]+=offset
            sample=reader.update(hypothetical)
            for contact in reader.data.contact:
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g))for g in(contact.geom1,contact.geom2)]
                if not contact_permitted(names,contact.dist,'approach'):
                    raise ValueError('NEW_WORKSPACE_SAVED_STARTUP_CONTACT:'+str((count,names,float(contact.dist))))
            if count==5703:first=sample
        if first is None or integrations:raise ValueError('FULL_5703_SAVED_STARTUP_ZERO_INTEGRATIONS')
    result=dict(schema='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1',
        status='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED',source_sha=spec['source_sha'],
        builder_sha256=digest(__file__),parent=scene_id(layout),split='TRAIN',layout=layout,
        geometry=geometry(layout),robot_contract=contract,saved_startup_pose_checks=5703,
        saved_startup_input_sha256=spec['saved_startup_journal_sha256'],
        compiled=dict(scene_sha256=digest(path),scene_xml_sha256=digest(xml),base_scene_sha256=BASE_SCENE,
            fresh_XML_compilation=True,original_robot_contract_exact=True),
        original_scene_file_changed=False,physics_integrations=0,new_transitions=0,
        native_execution_authorized=False,WBC_changes=False,physics_dt_s=.002)
    (output/'STATIC_RESULT.private.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def admission(spec):
    from .stable_task_scene_v1 import pinned
    proof=pinned(spec['task_learning_scene_reference'])
    layout=proof.get('layout')
    from . import task_learning_grasp_v1 as task
    expected_role=(task.QUALIFICATION_ROLE if layout in QUALIFICATION_OFFSETS else
        task.COLLECTION_ROLE if layout in TRAIN_OFFSETS else
        task.RECOVERY_QUALIFICATION_ROLE if layout in RECOVERY_OFFSETS else task.CONSTRUCTION_ROLE)
    if not(proof.get('schema')=='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1'
        and proof.get('status')=='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
        and proof.get('builder_sha256')==spec['source_files'][MEMBER]
        and proof.get('geometry')==geometry(layout)and proof.get('parent')==scene_id(layout)==spec['scene_id']==spec['data_parent']
        and spec.get('task_learning_role',task.CONSTRUCTION_ROLE)==expected_role
        and proof.get('split')==spec['data_role']=='TRAIN'
        and proof.get('saved_startup_pose_checks')==5703
        and proof.get('robot_contract',{}).get('status')=='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'
        and proof.get('compiled',{}).get('base_scene_sha256')==BASE_SCENE
        and proof['compiled'].get('fresh_XML_compilation')is True
        and proof['compiled'].get('original_robot_contract_exact')is True
        and proof['compiled'].get('scene_sha256')==digest(spec['model_path'])
        and proof.get('physics_integrations')==proof.get('new_transitions')==0
        and proof.get('native_execution_authorized')is False and proof.get('WBC_changes')is False
        and proof.get('physics_dt_s')==.002):raise ValueError('EXACT_NEW_NEAR_WORKSPACE_STATIC_PROVENANCE')
    return proof
