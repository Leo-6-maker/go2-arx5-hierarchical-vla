"""Independent current-state predicted-goal smooth TRAIN diagnostic admission.

Reuses the original registered serial workers, safety and command consumer.
This namespace cannot satisfy the old DEV or cross-hardware contracts.
"""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .storage import digest,write_json
from .stable_task_scene_v1 import pinned
from .task_learning_endpoint_policy_v3 import PROTOCOL,METHODS
from .task_learning_conic_policy_v3 import C500_PACKAGE_SHA256,verify_C500_package

SCHEMA='CORE_C500_ENDPOINT_BASIS_NATIVE_PANEL_V3'
INPUT_SCHEMA='CORE_C500_ENDPOINT_BASIS_GROUP_INPUT_V3'
NATIVE_SCHEMA='CORE_C500_ENDPOINT_BASIS_NATIVE_ACTUAL_V3'
EXPERIMENT='CORE_C500_ENDPOINT_BASIS_TRAIN_DIAGNOSTIC_V3'
PURPOSE='NEW_TRAIN_C500_ENDPOINT_BASIS_DIAGNOSTIC_NOT_BC_DEV_TEST'
CONTRACT_SHA='3e019ecb996f1e21dfe800ac562848ba6917976412df7ef18f80455adc5cdf93'
ORDER=(('TRAIN03',METHODS[0],2026111100),('TRAIN14',METHODS[0],2026111200))


def protocol():
    old=task.protocol(task.CORE_LOCAL_ROLE)
    return dict(old,stage=PROTOCOL,scene_family='TRAIN03_TRAIN14_ENDPOINT_BASIS_DIAGNOSTIC_V3',
        policy='FIXED_C500_CURRENT_STATE_PREDICTED_FINAL_GOAL_QUINTIC',
        raw_GR00T_not_promoted_by_decoding=True)


def qualification(request,checkpoint_files):
    task.need(request['preregistration']['sha256']==CONTRACT_SHA,'ENDPOINT_FIXED_PREREGISTRATION')
    contract=pinned(request['preregistration'])
    task.need(contract['requested']==2 and [(a['layout'],a['method'],a['query_seed_base'])for a in contract['assignments']]==list(ORDER),
        'ENDPOINT_FIXED_TWO_NO_RETRY')
    from .task_learning_conic_native_v3 import qualification as previous_qualification
    previous=pinned(request['prior_conic_admission'])
    task.need(previous['package']==request['package'],'ENDPOINT_SAME_PREVIOUS_C500_CUSTODY')
    custody=previous_qualification(previous,checkpoint_files)
    value=pinned(request['basis_study']);root=Path(request['basis_study']['path']).parent
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    task.need(request['basis_study']['sha256']==contract['required_completed_study_result_sha256']
        and lease['status']=='CLOSED'and lease['worker_exit_code']==0
        and lease['spec_sha256']==digest(root/'EXECUTION_SPEC.json')
        and value['status']=='COMPLETE_ACTUAL_ENDPOINT_BASIS540_AND_ORIGINAL_NATIVE_CLONES2'
        and value['offline_requested']==value['offline_completed']==value['offline_original_envelope_pass']==540
        and value['native_clones_requested']==value['native_clones_completed']==value['native_clones_pass']==2
        and value['actual_new_model_queries']==value['optimizer_updates']==value['physics_integrations']==0
        and value['native_execution_admission']is False
        and all(c['original_native_certificate_pass']and c['original_native_certificate_source_sha256']==contract['required_original_certificate_sha256']for c in value['native_clones']),
        'ENDPOINT_REAL_CLOSED_540_AND_TWO_ORIGINAL_NATIVE_CERTIFICATES_NOT_SOFTWARE_FIXTURES')
    return custody


def policy_contract(value,checkpoint_files,*,verify_package=True,expected_global_step=2000):
    from .target_gr00t_profile_v1 import PARENT
    task.need(set(value)=={'arm','parent_manifest_sha256','package','Core_endpoint_qualification'}
        and value['arm']in METHODS and value['parent_manifest_sha256']==PARENT
        and value['package']['sha256']==C500_PACKAGE_SHA256 and expected_global_step==2000,
        'ENDPOINT_SEPARATE_C500_POLICY_NOT_OLD_B0_SFT_OR_FULL_SFT2500')
    q=pinned(value['Core_endpoint_qualification'])
    task.need(q['schema']=='CORE_C500_ENDPOINT_BASIS_ADMISSION_V3'and q['admission_class']==PROTOCOL
        and q['preregistration']['sha256']==CONTRACT_SHA and q['package']==value['package']
        and q['original_cross_hardware_REJECT_preserved']is True
        and q['original_DEV_TEST_admission']is False,'ENDPOINT_INDEPENDENT_LOCAL_ADMISSION')
    if verify_package:qualification(q,checkpoint_files)
    return q


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==PROTOCOL
        and panel.get('protocol')==protocol()and panel.get('B1_global_step')==2000
        and panel.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and set(panel.get('policies',{}))==set(METHODS)and len(rows)==2
        and len({r['native_run']for r in rows})==2,'ENDPOINT_INDEPENDENT_FIXED_TWO')
    for row,(layout,method,seed)in zip(rows,ORDER):
        task.need(row['layout']==layout and row['method']==method and row['parent']==scene_id(layout)
            and row['split']=='TRAIN'and row['purpose']==PURPOSE and row['restart']==1
            and row['command']==task.COMMAND and row['query_seeds']==list(range(seed,seed+15))
            and Path(row['native_run']).name==row['native_run']and len(row['model_sha256'])==64
            and panel['policies'][method]['arm']==method,'ENDPOINT_NO_NEW_NOISE_OR_LAYOUT_OR_RETRY')
    return rows


def selection(inp,panel):
    rows=validate_panel(panel)
    task.need(inp.get('schema')==INPUT_SCHEMA and inp.get('assignment_slice')==[0,2]
        and inp.get('population_predecessor')is None and inp.get('source_sha')==panel['source_sha'],
        'ENDPOINT_ONE_INDEPENDENT_COMPLETE_PANEL')
    return rows


def summarize(panel,outcomes):
    rows=validate_panel(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)
        and all(r['native_run']in wanted and all(r[k]==wanted[r['native_run']][k]
            for k in('layout','method','restart','parent','split','query_seeds'))for r in outcomes),'ENDPOINT_NO_FOREIGN_OR_RETRIED_OUTCOME')
    methods={}
    for method in METHODS:
        values=[r for r in outcomes if r['method']==method]
        methods[method]=dict(requested=2,attempted=sum(r['attempted']for r in values),
            task_successes=sum(r['task_success']and r['technical_valid']for r in values),
            valid_task_failures=sum(r['technical_valid']and not r['task_success']for r in values),
            technical_failures=sum(not r['technical_valid']for r in values),
            hard_safety_stop_conditions=sum(r['hard_safety_stop']for r in values),
            actual_GR00T_queries=sum(r.get('actual_GR00T_queries',0)for r in values),
            actual_native_integrations=sum(r.get('actual_native_integrations',0)for r in values))
    return dict(schema='CORE_C500_ENDPOINT_BASIS_NATIVE_SUMMARY_V3',requested=2,completed=len(outcomes),
        complete=len(outcomes)==2,methods=methods,original_DEV_conditions=0,TEST_conditions=0,
        qualified_training_demonstrations=0,model_updates=0,generalization_claim=False,
        original_cross_hardware_REJECT_replaced=False)


def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry,scene_id,BASE_SCENE
    from .task_learning_local_policy_v1 import scene_builder_custody
    proof=pinned(spec['task_learning_scene_reference']);layout=proof['layout']
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    builder=scene_builder_custody(panel['registration']['scene_producers'][layout],
        spec['task_learning_scene_reference'],spec['model_path'])
    task.need(layout in('TRAIN03','TRAIN14')and spec.get('core_local_protocol')==PROTOCOL
        and spec.get('task_learning_role')==PROTOCOL and spec.get('data_role')=='TRAIN'
        and proof['schema']=='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1'
        and proof['status']=='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
        and proof['builder_sha256']==builder and proof['geometry']==geometry(layout)
        and proof['parent']==scene_id(layout)==spec['scene_id']==spec['data_parent']
        and proof['split']=='TRAIN'and proof['saved_startup_pose_checks']==5703
        and proof['robot_contract']['status']=='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'
        and proof['compiled']['base_scene_sha256']==BASE_SCENE
        and proof['compiled']['fresh_XML_compilation']is True and proof['compiled']['original_robot_contract_exact']is True
        and proof['compiled']['scene_sha256']==digest(spec['model_path'])
        and proof['physics_integrations']==proof['new_transitions']==0
        and proof['native_execution_authorized']is False and proof['WBC_changes']is False
        and proof['physics_dt_s']==.002,'ENDPOINT_SAME_SEALED_TRAIN_SCENE_DYNAMICS_AND_SAFETY')
    matches=[r for r in validate_panel(panel)if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'ENDPOINT_ONE_FIXED_NATIVE_REQUEST');row=matches[0]
    task.need(row['parent']==proof['parent']and row['model_sha256']==digest(spec['model_path'])
        and spec.get('target_gr00t_policy')==panel['policies'][row['method']]
        and spec.get('target_gr00t_query_seeds')==row['query_seeds']
        and spec.get('target_gr00t_pair_id')==row['native_run']
        and spec.get('task_learning_teacher_mode')is None and spec.get('task_learning_control')is None,
        'ENDPOINT_EXACT_C500_METHOD_AND_NO_TEACHER_FALLBACK')
    return proof


def validate_live(s,*,verify_model=True):
    from scripts.pi0292.relay_protocol import digest as document_hash
    from scripts.pi0294.numerics import MODE
    from .fusion_paused_inference import CHECKPOINT_INVENTORY
    from .target_gr00t_profile_v1 import PARENT
    from .task_learning_live_pair_v1 import memory_contract,PROFILE,KIND
    task.need(s.get('core_local_protocol')==PROTOCOL and s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL'and s.get('physics')is False
        and s.get('device')=='cuda'and s.get('max_queries')==15
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed')is False
        and s.get('checkpoint_manifest_sha256')==PARENT and s.get('task_learning_grasp')==protocol()
        and document_hash(s['checkpoint_files'])==CHECKPOINT_INVENTORY,'ENDPOINT_ORIGINAL_NUMERICS_AND_MODEL_INVENTORY')
    descriptor=s['producer_native_spec'];producer=pinned(descriptor);task.declared(producer)
    task.need(Path(descriptor['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('task_learning_role')==PROTOCOL and producer.get('core_local_protocol')==PROTOCOL
        and producer.get('target_gr00t_gpu_root')==s['registered_root']and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and producer.get('target_gr00t_pair_id')==s.get('target_gr00t_pair_id')
        and producer.get('target_gr00t_B1_global_step')==s.get('target_gr00t_B1_global_step')==2000,
        'ENDPOINT_EXACT_CURRENT_NATIVE_AND_NEURAL_PAIR')
    native_scene_admission(producer)
    q=policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model)
    task.need(s.get('GPU_uuid')==q['GPU_uuid'],'ENDPOINT_SAME_QUALIFIED_CORE_CARD')
    memory_contract(s);return producer


def load(s,root):
    """The CPU install then one CUDA move used by the actual parity2 loader."""
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen
    from scripts.pi0294.numerics import MODE
    from .task_learning_conic_policy_v3 import install_ordinary_model_delta
    from .target_gr00t_core_v1 import trim_unused_logits
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    holder={};package=s['target_gr00t_policy']['package']
    def initialize(model):
        holder['custody']=install_ordinary_model_delta(model,package,s['checkpoint_files'])
        return dict(package=package,ordinary_CPU_install=True)
    instance,torch=_load_frozen(s,root,cpu_model_initializer=initialize);trim_unused_logits(instance.model)
    custody=holder['custody']
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    instance.processor=Gr00tN1d7Processor.from_pretrained(Path(custody['parent_package']['path'])/'processor',
        model_name=local_model_name(s['registered_root'],s['neural_bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['neural_bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    instance.processor.eval();instance.identity=dict(package=package,parent_package=custody['parent_package'],numeric_mode=MODE)
    write_json(Path(root)/'TARGET_POLICY_LOADED.private.json',dict(identity=instance.identity,
        actual_model_loads=1,ordinary_internal_GR00T_weights=True,external_inference_adapter=False,
        explicit_policy_method=s['target_gr00t_policy']['arm']))
    return instance,torch


def register_panel(request):
    """Use the existing owner/Journal/registry; seal two new diagnostic requests."""
    import os,time
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .task_learning_student_registration_v2 import closed_base,actual_site
    from .task_learning_scene_v1 import scene_id
    from .task_learning_local_policy_v1 import scene_builder_custody
    n=closed_base(request['native_base']);g=closed_base(request['gpu_base'])
    campaign=Path(n['campaign_root']);actual_site(campaign)
    source=Path(request['source_root']);root=Path(request['root']);q=pinned(request['Core_endpoint_qualification'])
    task.need(root.parent==Path(n['registered_root']).parents[2]and source.resolve()==source,
              'LOCAL_PANEL_UNDER_ORIGINAL_PROJECT_DATA_ROOT')
    task.need(all(digest(source/name)==pin for name,pin in request['source_files'].items()),
              'SEALED_LOCAL_RUNTIME_SOURCE')
    from .target_gr00t_profile_v1 import PARENT
    policies={m:dict(arm=m,parent_manifest_sha256=PARENT,package=q['package'],
                    Core_endpoint_qualification=request['Core_endpoint_qualification'])for m in METHODS}
    for m in policies:policy_contract(policies[m],g['checkpoint_files'])
    rows=[]
    for index,(layout,method,seed)in enumerate(ORDER):
        descriptor=request['scenes'][layout];proof=pinned(descriptor)
        model=Path(descriptor['path']).parent/'SCENE.private.mjb'
        scene_builder_custody(request['scene_producers'][layout],descriptor,model)
        task.need(proof['layout']==layout and proof['split']=='TRAIN'
            and proof['compiled']['scene_sha256']==digest(model),'PREREGISTERED_TRAIN_SCENE_NOT_DEV_TEST')
        generation=root.name.rsplit('-',1)[-1]
        task.need(generation.isdigit(),'UNIQUE_LOCAL_REPAIRED_PANEL_GENERATION')
        rows.append(dict(native_run=f'tl-core-endpoint-20261111-v{generation}-{index:02d}-{method.lower()}',method=method,
            layout=layout,parent=scene_id(layout),restart=1,split='TRAIN',purpose=PURPOSE,
            command=task.COMMAND,model_sha256=digest(model),query_seeds=list(range(seed,seed+15))))
    panel=dict(schema=SCHEMA,task_learning_role=PROTOCOL,protocol=task.protocol(PROTOCOL),
        source_sha=request['source_sha'],B1_global_step=2000,policies=policies,assignments=rows,
        preregistration=request['preregistration'],original_DEV_TEST_modified=False,
        registration=dict(native_base=request['native_base'],gpu_base=request['gpu_base'],
            source_root=str(source),source_files=request['source_files'],scenes=request['scenes'],scene_producers=request['scene_producers'],
            scene_allowlist=n['scene_allowlist'],scene_allowlist_sha256=n['scene_allowlist_sha256'],
            memory_execution=g['memory_execution'],memory_execution_reference=g['memory_execution_reference']))
    validate_panel(panel);key=root.name.upper()+'-PANEL';journal=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists()and not journal.path(key).exists(),'UNKNOWN_LOCAL_PANEL_NEVER_RECREATE')
        def effect():
            root.mkdir();(root/'control').mkdir();write(root/'PANEL.private.json',panel)
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',planned_conditions=2,
                panel_sha256=digest(root/'PANEL.private.json'),created_epoch=time.time()))
            reg,pin=snapshot(n['registry'])
            task.need(not any(e.get('path')==str(root)for e in reg['entries']),'ONE_LOCAL_PANEL_REGISTRY')
            reg['entries'].append(dict(path=str(root),role=key,scope=SCHEMA,owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],reg,key+'-REGISTRY',expected_before_sha256=pin)
            return dict(status='CORE_C500_ENDPOINT_BASIS_TWO_CONDITIONS_SEALED',panel=dict(path=str(root/'PANEL.private.json'),sha256=digest(root/'PANEL.private.json')))
        return journal.once(key,dict(source_sha=request['source_sha'],qualification=request['Core_endpoint_qualification']),effect)
