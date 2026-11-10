"""Typed Core-local admission and four TRAIN-scene diagnostic conditions.

This proof is rejected by the original cross-hardware/DEV policy contract.
Admission permits an attempt through the unchanged native rejection gates.
It never licenses a rejected proposal or promotes a diagnostic to DEV/TEST.
"""
import json
from pathlib import Path

import numpy as np

from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest,write_json
from .target_gr00t_profile_v1 import PARENT

PROTOCOL=task.CORE_LOCAL_ROLE
SCHEMA='TASK_LEARNING_CORE_LOCAL_PANEL_V1'
INPUT_SCHEMA='TASK_LEARNING_CORE_LOCAL_GROUP_INPUT_V1'
NATIVE_SCHEMA='TASK_LEARNING_CORE_LOCAL_NATIVE_ACTUAL_V1'
EXPERIMENT='CORE_LOCAL_B0_SFT2000_DIAGNOSTIC_NOT_DEV_TEST'
PURPOSE='CORE_LOCAL_DIAGNOSTIC_NOT_BC_DEV_OR_TEST'
CONTRACT_SHA='86ac376bc7f06f592333a2e635d7f6877dec7ecf2d1003848fc58dd41bb2f506'
ORDER=(('TRAIN01','B0',20261110100),('TRAIN01','B1',20261110100),
       ('TRAIN13','B1',20261110200),('TRAIN13','B0',20261110200))
EXECUTION_STAGE_REVISION='CORE_LOCAL_GENERATED_GRIP_CONTACT_INTENT_V2'


def execution_stage(raw,sample):
    """Supply the original contact predicate's intent; never edit its rules.

    Actual grasp is native mechanical feedback, not a learner observation.
    An opening proposal while already grasping stays in approach and can be
    rejected by the original certificate; no attachment hypothesis is invented.
    """
    raw=np.asarray(raw)
    task.need(raw.shape==(40,10)and np.isfinite(raw).all()
        and np.all((raw[:,9]>=0)&(raw[:,9]<=1)),'UNCHANGED_DECODED_GRIP_FOR_EXECUTION_CLASS')
    if sample.get('native_grasp')is True and np.all(raw[:,9]<=.5):return 'HOLD'
    if raw[-1,9]<=.5 and np.min(raw[:,9])<.5:return 'CLOSE'
    return 'PREGRASP'


def replay(pin,arm,package,GPU_uuid):
    result=pinned(pin);root=Path(pin['path']).parent
    spec_path=root/'EXECUTION_SPEC.json';spec=json.loads(spec_path.read_bytes())
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    same=json.loads((root/'SAME_CARD_ACTUAL.private.json').read_bytes())
    task.need(Path(pin['path'])==root/'LOCAL_REPLAY_ACTUAL.private.json'
        and result.get('schema')=='TASK_LEARNING_CORE_LOCAL_ORDINARY_REPLAY_ACTUAL_V1'
        and result.get('policy_arm')==arm and result.get('same_card_reproducible')is True
        and result.get('actual_queries')==(80 if arm=='B0'else 40)
        and result.get('GPU_uuid')==GPU_uuid and result.get('denoising_steps')==4
        and result.get('precision')=='ORIGINAL_BF16_AUTOCAST_FP32_PARAMETERS'
        and result.get('original_cross_hardware_REJECT_preserved')is True
        and result.get('mechanical_safety_or_consumption_audit_changed')is False
        and result.get('clipping')is False and result.get('Teacher_fallback')is False
        and result.get('optimizer_updates')==result.get('native_integrations')==0
        and lease.get('status')=='CLOSED'and lease.get('spec_sha256')==digest(spec_path)
        and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('actual_model_loads')==1 and worker.get('failure')is None
        and spec.get('operation')=='LOCAL_REPLAY'and spec.get('policy_arm')==arm
        and not any(spec.get(k)for k in('diagnostic_only','tensor_trace','backbone_trace','inference_precision_diagnostic','numerical_diagnostic'))
        and same.get('schema')=='TASK_LEARNING_CORE_LOCAL_SAME_CARD_REPLAY_V1'
        and same.get('requested')==same.get('matched')==len(same.get('records',[]))==40
        and same.get('all_bit_exact')is True
        and all(all(r['exact'].values())for r in same['records']),
        'ACTUAL_SEPARATE_ORDINARY_SAME_CARD_REPLAY_NOT_CROSS_HARDWARE_DIAGNOSTIC')
    task.need(spec.get('package')==package and spec.get('parent',{}).get('files'),
              'PINNED_ORIGINAL_PARENT_AND_SFT2000_PACKAGE')
    measurements=['replay-1','replay-2']if arm=='B0'else['replay-1']
    for folder in measurements:
        measured=json.loads((root/folder/'RESULT.private.json').read_bytes())
        task.need(measured.get('actual_queries')==len(measured.get('records',[]))==40
            and measured.get('original_full_inference')is True and measured.get('adaptation_step')==0,
            'ALL40_ORDINARY_FULL_INFERENCE_NOT_ACTION_HEAD_FEATURE_SWAP')
        for row in measured['records']:
            file=root/folder/row['sample_file']
            task.need(Path(row['sample_file']).name==row['sample_file']and digest(file)==row['sha256'],
                      'PINNED_LOCAL_SEMANTICS_PREDICTION')
            with np.load(file,allow_pickle=False)as z:
                raw=z['physical_H40'];normalized=z['normalized']
                task.need(raw.shape==(40,10)and normalized.shape==(1,40,132)
                    and np.isfinite(raw).all()and np.isfinite(normalized).all()
                    and np.all((raw[:,9]>=0)&(raw[:,9]<=1)),
                    'FINITE_ABSOLUTE_EEF9_GRIP1_FULL_H40_ORIGINAL_SINGLE_DECODE')
    return dict(result=pin,spec=dict(path=str(spec_path),sha256=digest(spec_path)),
                same_card=dict(path=str(root/'SAME_CARD_ACTUAL.private.json'),sha256=digest(root/'SAME_CARD_ACTUAL.private.json')))


def create_admission(request,output):
    task.need(request['preregistration']['sha256']==CONTRACT_SHA and pinned(request['preregistration'])['Core_local_native']['requested_conditions']==4,
              'FROZEN_FOUR_CONDITION_LOCAL_CONTRACT_BEFORE_NATIVE_RESULTS')
    actual={arm:replay(request['replays'][arm],arm,request['package'],request['GPU_uuid'])for arm in('B0','B1')}
    value=dict(schema='TASK_LEARNING_CORE_LOCAL_POLICY_ADMISSION_V1',admission_class=PROTOCOL,
        package=request['package'],parent_manifest_sha256=PARENT,GPU_uuid=request['GPU_uuid'],
        precision='ORIGINAL_BF16_AUTOCAST_FP32_PARAMETERS',replays=actual,
        preregistration=request['preregistration'],allowed_panel_contract_sha256=CONTRACT_SHA,
        original_cross_hardware_REJECT_preserved=True,cross_hardware_qualification=False,
        original_DEV_TEST_admission=False,weights_or_Processor_changed_by_admission=False,
        qualified_for='FOUR_TRAIN_SCENE_DIAGNOSTIC_ATTEMPTS_THROUGH_ORIGINAL_NATIVE_GATES',
        unsafe_proposal='REJECT_BEFORE_EXECUTION_NO_CLIP_NO_TEACHER_FALLBACK',
        execution_stage_revision=EXECUTION_STAGE_REVISION,
        new_model_calls=0,optimizer_updates=0,physics_integrations=0)
    write_json(output,value);return value


def policy_contract(value,checkpoint_files,*,verify_package=True,expected_global_step=2000):
    task.need(set(value)=={'arm','parent_manifest_sha256','package','Core_local_qualification'}
        and value['arm']in('B0','B1')and value['parent_manifest_sha256']==PARENT
        and expected_global_step==2000,'INDEPENDENT_LOCAL_B0_OR_FROZEN_SFT2000_ONLY')
    q=pinned(value['Core_local_qualification'])
    task.need(q.get('schema')=='TASK_LEARNING_CORE_LOCAL_POLICY_ADMISSION_V1'
        and q.get('admission_class')==PROTOCOL and q.get('allowed_panel_contract_sha256')==CONTRACT_SHA
        and q.get('original_cross_hardware_REJECT_preserved')is True
        and q.get('cross_hardware_qualification')is False and q.get('original_DEV_TEST_admission')is False,
        'LOCAL_ADMISSION_NEVER_ORIGINAL_NUMERICAL_GATE_OR_DEV_PROMOTION')
    task.need(q.get('execution_stage_revision')==EXECUTION_STAGE_REVISION,
              'CORRECTED_CONTACT_INTENT_BRIDGE_WITH_ORIGINAL_PREDICATE')
    task.need((value['arm']=='B0'and value['package']is None)
        or(value['arm']=='B1'and value['package']==q['package']),
        'B0_ORIGINAL_WEIGHTS_PROCESSOR_OR_ONE_FROZEN_SFT_PACKAGE')
    record=q['replays'][value['arm']]
    spec=pinned(record['spec']);result=pinned(record['result']);same=pinned(record['same_card'])
    task.need(spec['policy_arm']==value['arm']and result['same_card_reproducible']is True
        and same['all_bit_exact']is True and result['GPU_uuid']==q['GPU_uuid'],
        'ACTUAL_REPLAY_CUSTODY_REMAINS_PINNED')
    if verify_package and value['arm']=='B1':
        from .target_gr00t_delivery_v1 import verify
        m=verify(value['package']['path'],value['package']['sha256'],checkpoint_files)
        task.need(m['global_step']==2000 and m['recipe']['variant']=='FULL_ACTION_SIDE',
                  'ORIGINAL_SFT2000_B1_NOT_NEW_ADAPTATION_OR_FIT4')
    return q


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==PROTOCOL
        and panel.get('protocol')==task.protocol(PROTOCOL)and panel.get('B1_global_step')==2000
        and panel.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and set(panel.get('policies',{}))=={'B0','B1'}and len(rows)==4
        and len({r['native_run']for r in rows})==4,'EXACT_SEPARATE_LOCAL_FOUR_NOT_DEV80')
    for row,(layout,method,seed)in zip(rows,ORDER):
        task.need(row['layout']==layout and row['method']==method and row['parent']==scene_id(layout)
            and row['split']=='TRAIN'and row['purpose']==PURPOSE and row['restart']==1
            and row['command']==task.COMMAND and row['query_seeds']==list(range(seed,seed+15))
            and Path(row['native_run']).name==row['native_run']and len(row['model_sha256'])==64,
            'PREREGISTERED_METHOD_LAYOUT_RESTART_NOISE_NO_DEV_TEST_OR_RETRY')
    return rows


def selection(inp,panel):
    rows=validate_panel(panel)
    task.need(inp.get('schema')==INPUT_SCHEMA and inp.get('assignment_slice')==[0,4]
        and inp.get('population_predecessor')is None and inp.get('source_sha')==panel['source_sha'],
        'ONE_FULL_FOUR_CONDITION_LOCAL_PANEL')
    return rows


def summarize(panel,outcomes):
    rows=validate_panel(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({v['native_run']for v in outcomes})==len(outcomes)
        and all(v['native_run']in wanted and all(v[k]==wanted[v['native_run']][k]
            for k in('layout','method','restart','parent','query_seeds'))for v in outcomes),
        'NO_LOCAL_RETRY_FOREIGN_CASE_OR_RELABELED_DEV')
    methods={}
    for method in('B0','B1'):
        values=[r for r in outcomes if r['method']==method]
        methods[method]=dict(requested=2,attempted=sum(r['attempted']for r in values),
            task_successes=sum(r['task_success']and r['technical_valid']for r in values),
            valid_task_failures=sum(r['technical_valid']and not r['task_success']for r in values),
            technical_failures=sum(not r['technical_valid']for r in values),
            hard_safety_stop_conditions=sum(r['hard_safety_stop']for r in values),
            actual_GR00T_queries=sum(r.get('actual_GR00T_queries',0)for r in values))
    return dict(schema='TASK_LEARNING_CORE_LOCAL_SUMMARY_V1',requested=4,completed=len(outcomes),
        complete=len(outcomes)==4,methods=methods,original_DEV_conditions=0,TEST_conditions=0,
        qualified_training_demonstrations=0,model_updates=0,generalization_claim=False,
        original_cross_hardware_REJECT_replaced=False)


def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry,scene_id,BASE_SCENE,MEMBER
    proof=pinned(spec['task_learning_scene_reference']);layout=proof['layout']
    task.need(layout in('TRAIN01','TRAIN13')and spec.get('core_local_protocol')==PROTOCOL
        and spec.get('data_role')=='TRAIN'and spec.get('task_learning_role')==PROTOCOL
        and proof['schema']=='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1'
        and proof['status']=='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
        and proof['builder_sha256']==spec['source_files'][MEMBER]
        and proof['geometry']==geometry(layout)and proof['parent']==scene_id(layout)==spec['scene_id']==spec['data_parent']
        and proof['split']=='TRAIN'and proof['saved_startup_pose_checks']==5703
        and proof['robot_contract']['status']=='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'
        and proof['compiled']['base_scene_sha256']==BASE_SCENE
        and proof['compiled']['fresh_XML_compilation']is True and proof['compiled']['original_robot_contract_exact']is True
        and proof['compiled']['scene_sha256']==digest(spec['model_path'])
        and proof['physics_integrations']==proof['new_transitions']==0
        and proof['native_execution_authorized']is False and proof['WBC_changes']is False
        and proof['physics_dt_s']==.002,'UNCHANGED_STATIC_TRAIN_SCENE_FOR_LOCAL_DIAGNOSTIC_ONLY')
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    rows=validate_panel(panel);matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'ONE_PREREGISTERED_LOCAL_REQUEST');row=matches[0]
    task.need(row['parent']==proof['parent']and row['model_sha256']==digest(spec['model_path'])
        and spec.get('target_gr00t_policy')==panel['policies'][row['method']]
        and spec.get('target_gr00t_query_seeds')==row['query_seeds']
        and spec.get('target_gr00t_pair_id')==row['native_run']
        and spec.get('task_learning_teacher_mode')is None and spec.get('task_learning_control')is None,
        'EXACT_FROZEN_LOCAL_METHOD_NO_TEACHER_FALLBACK')
    return proof


def validate_live(s,*,verify_model=True):
    from scripts.pi0292.relay_protocol import digest as document_hash
    from scripts.pi0294.numerics import MODE
    from .fusion_paused_inference import CHECKPOINT_INVENTORY
    from .task_learning_live_pair_v1 import memory_contract,PROFILE,KIND
    task.need(s.get('core_local_protocol')==PROTOCOL and s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL'and s.get('physics')is False
        and s.get('device')=='cuda'and s.get('max_queries')==15
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed')is False
        and s.get('checkpoint_manifest_sha256')==PARENT and s.get('task_learning_grasp')==task.protocol(PROTOCOL)
        and document_hash(s['checkpoint_files'])==CHECKPOINT_INVENTORY,'EXACT_LOCAL_ORIGINAL_RUNTIME_AND_INVENTORY')
    descriptor=s['producer_native_spec'];producer=pinned(descriptor);task.declared(producer)
    task.need(Path(descriptor['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('task_learning_role')==PROTOCOL and producer.get('core_local_protocol')==PROTOCOL
        and producer.get('target_gr00t_gpu_root')==s['registered_root']and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and producer.get('target_gr00t_pair_id')==s.get('target_gr00t_pair_id')
        and type(s.get('target_gr00t_pair_id'))is str and s['target_gr00t_pair_id']
        and producer.get('target_gr00t_B1_global_step')==s.get('target_gr00t_B1_global_step')==2000,
        'SAME_LOCAL_NATIVE_NEURAL_PAIR_NOT_ORIGINAL_DEV')
    native_scene_admission(producer)
    q=policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model)
    task.need(s.get('GPU_uuid')==q['GPU_uuid'],'SAME_QUALIFIED_CORE_CARD')
    memory_contract(s)
    return producer


def register_panel(request):
    """Use the existing owner/Journal/registry; seal four new diagnostic requests."""
    import os,time
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .task_learning_student_registration_v2 import closed_base,actual_site
    from .task_learning_scene_v1 import scene_id
    n=closed_base(request['native_base']);g=closed_base(request['gpu_base'])
    campaign=Path(n['campaign_root']);actual_site(campaign)
    source=Path(request['source_root']);root=Path(request['root']);q=pinned(request['Core_local_qualification'])
    task.need(root.parent==Path(n['registered_root']).parents[2]and source.resolve()==source,
              'LOCAL_PANEL_UNDER_ORIGINAL_PROJECT_DATA_ROOT')
    task.need(all(digest(source/name)==pin for name,pin in request['source_files'].items()),
              'SEALED_LOCAL_RUNTIME_SOURCE')
    policies={m:dict(arm=m,parent_manifest_sha256=PARENT,package=q['package']if m=='B1'else None,
                    Core_local_qualification=request['Core_local_qualification'])for m in('B0','B1')}
    for m in policies:policy_contract(policies[m],g['checkpoint_files'])
    rows=[]
    for index,(layout,method,seed)in enumerate(ORDER):
        descriptor=request['scenes'][layout];proof=pinned(descriptor)
        model=Path(descriptor['path']).parent/'SCENE.private.mjb'
        task.need(proof['layout']==layout and proof['split']=='TRAIN'
            and proof['compiled']['scene_sha256']==digest(model),'PREREGISTERED_TRAIN_SCENE_NOT_DEV_TEST')
        rows.append(dict(native_run=f'tl-core-local-20261110-{index:02d}-{method.lower()}',method=method,
            layout=layout,parent=scene_id(layout),restart=1,split='TRAIN',purpose=PURPOSE,
            command=task.COMMAND,model_sha256=digest(model),query_seeds=list(range(seed,seed+15))))
    panel=dict(schema=SCHEMA,task_learning_role=PROTOCOL,protocol=task.protocol(PROTOCOL),
        source_sha=request['source_sha'],B1_global_step=2000,policies=policies,assignments=rows,
        preregistration=request['preregistration'],original_DEV_TEST_modified=False,
        registration=dict(native_base=request['native_base'],gpu_base=request['gpu_base'],
            source_root=str(source),source_files=request['source_files'],scenes=request['scenes'],
            scene_allowlist=n['scene_allowlist'],scene_allowlist_sha256=n['scene_allowlist_sha256'],
            memory_execution=g['memory_execution'],memory_execution_reference=g['memory_execution_reference']))
    validate_panel(panel);key=root.name.upper()+'-PANEL';journal=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists()and not journal.path(key).exists(),'UNKNOWN_LOCAL_PANEL_NEVER_RECREATE')
        def effect():
            root.mkdir();(root/'control').mkdir();write(root/'PANEL.private.json',panel)
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',planned_conditions=4,
                panel_sha256=digest(root/'PANEL.private.json'),created_epoch=time.time()))
            reg,pin=snapshot(n['registry'])
            task.need(not any(e.get('path')==str(root)for e in reg['entries']),'ONE_LOCAL_PANEL_REGISTRY')
            reg['entries'].append(dict(path=str(root),role=key,scope=SCHEMA,owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],reg,key+'-REGISTRY',expected_before_sha256=pin)
            return dict(status='CORE_LOCAL_FOUR_CONDITIONS_SEALED',panel=dict(path=str(root/'PANEL.private.json'),sha256=digest(root/'PANEL.private.json')))
        return journal.once(key,dict(source_sha=request['source_sha'],qualification=request['Core_local_qualification']),effect)
