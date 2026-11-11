"""Fixed matched weights/Processor/representation TRAIN development comparison.

Reuses the original registered serial workers, safety and command consumer.
This namespace cannot satisfy the old DEV or cross-hardware contracts.
"""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .storage import digest,write_json
from .stable_task_scene_v1 import pinned
from .task_learning_matched_policy_v3_1 import PROTOCOL,METHODS,configuration
from .task_learning_conic_policy_v3 import C500_PACKAGE_SHA256,verify_C500_package

SCHEMA='CORE_MATCHED128_SMOOTH_NATIVE_PANEL_V3'
INPUT_SCHEMA='CORE_MATCHED128_SMOOTH_GROUP_INPUT_V3'
NATIVE_SCHEMA='CORE_MATCHED128_SMOOTH_NATIVE_ACTUAL_V3'
EXPERIMENT='CORE_MATCHED128_SMOOTH_TRAIN_DIAGNOSTIC_V3'
PURPOSE='MATCHED128_TRAIN_DEVELOPMENT_NOT_BC_DEV_TEST'
CONTRACT_SHA='f91a57f07c18373e8dc9addd16028e76af538fd7cdee6aeb273deca54b892e62'
LAYOUTS=('TRAIN02','TRAIN05','TRAIN08','TRAIN11','TRAIN15','TRAIN18','TRAIN21','TRAIN24')


def order():
    result=[]
    for index,layout in enumerate(LAYOUTS):
        for restart in (1,2):
            rotation=(2*index+restart-1)%len(METHODS)
            for method in METHODS[rotation:]+METHODS[:rotation]:
                result.append((layout,restart,method,202611110000+1000*index+100*restart))
    return tuple(result)

ORDER=order()
_QUALIFIED_IMMUTABLE_INPUTS={}



def protocol():
    old=task.protocol(task.CORE_LOCAL_ROLE)
    return dict(old,stage=PROTOCOL,scene_family='EIGHT_FIXED_TRAIN_LAYOUTS_MATCHED_CONFIGURATIONS_V3_1',
        policy='FOUR_FIXED_WEIGHT_PROCESSOR_CONFIGURATIONS_EACH_RAW_OR_SAME_QUINTIC',
        raw_GR00T_not_promoted_by_decoding=True)


def closed_result(reference):
    value=pinned(reference);root=Path(reference['path']).parent
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    task.need(lease['status']=='CLOSED'and lease['worker_exit_code']==0,
        'MATCHED_ACTUAL_CLOSED0_EVIDENCE')
    source=root/('INPUT.private.json'if 'input_sha256'in lease else'EXECUTION_SPEC.json')
    task.need(digest(source)==lease.get('input_sha256',lease.get('spec_sha256')),
        'MATCHED_ORIGINAL_PRODUCER_BINDING')
    return value


def qualification(request,checkpoint_files):
    # All inputs are already closed and hash-bound. Validate once per process
    # at this new version boundary, rather than reread large parent packages
    # for each of the eight method metadata records. Actual model loaders still
    # check the weights they load; mutable scopes and live state are not cached.
    keys=('preregistration','local_admission','endpoint_admission','SFT2000_package',
          'C500_package','B0_target_replay','native_pilot','GPU_uuid')
    key=json.dumps([{name:request[name]for name in keys},checkpoint_files],sort_keys=True)
    if key not in _QUALIFIED_IMMUTABLE_INPUTS:
        _QUALIFIED_IMMUTABLE_INPUTS[key]=_qualification(request,checkpoint_files)
    return _QUALIFIED_IMMUTABLE_INPUTS[key]


def _qualification(request,checkpoint_files):
    from .target_gr00t_profile_v1 import PARENT
    from .task_learning_local_policy_v1 import policy_contract as local_contract
    from .task_learning_endpoint_native_v3 import policy_contract as endpoint_contract
    task.need(request['preregistration']['sha256']==CONTRACT_SHA,'MATCHED_FROZEN128_PREREGISTRATION')
    contract=pinned(request['preregistration'])
    task.need(contract['layouts']==list(LAYOUTS)and contract['methods']==list(METHODS)
        and contract['requested_conditions']==128 and contract['conditions_per_method']==16,
        'MATCHED_SAME128_POPULATION')
    for arm in ['B0','B1']:
        local_contract(dict(arm=arm,parent_manifest_sha256=PARENT,package=request['SFT2000_package']if arm=='B1'else None,
            Core_local_qualification=request['local_admission']),checkpoint_files)
    custody=endpoint_contract(dict(arm='C500_PLUS_ENDPOINT_BASIS',parent_manifest_sha256=PARENT,
        package=request['C500_package'],Core_endpoint_qualification=request['endpoint_admission']),checkpoint_files)
    task.need(request['SFT2000_package']['sha256']=='7feffa8dd11b5c9ce963f2e0f2f2613196c1a3ae8e5039056377bd5e25e02250'
        and request['C500_package']['sha256']==C500_PACKAGE_SHA256,'MATCHED_NO_NEW_WEIGHT_GENERATION')
    result=closed_result(request['B0_target_replay']);root=Path(request['B0_target_replay']['path']).parent
    same=json.loads((root/'SAME_CARD_ACTUAL.private.json').read_bytes())
    task.need(result['status']=='PASS_ACTUAL_NEW_B0_TARGET_PROCESSOR_SAME_CARD40_TWICE'
        and result['actual_queries']==result['requested_queries']==80 and result['actual_model_loads']==1
        and result['matched_unique_states']==len(same['records'])==40 and result['all_bit_exact']and same['all_bit_exact']
        and all(all(row['exact'].values())for row in same['records'])
        and result['checkpoint_manifest_sha256']==PARENT and result['processor_package']==request['SFT2000_package']
        and result['original_B0_weights_unchanged']and not result['SFT_or_LAST2_weights_installed']
        and result['optimizer_updates']==result['physics_integrations']==0
        and result['GPU_uuid']==request['GPU_uuid'],'MATCHED_REUSE_REAL_NEW_B0_TARGET80')
    for repeat in [1,2]:
        folder=root/f'replay-{repeat}';records=json.loads((folder/'RESULT.private.json').read_bytes())['records']
        task.need(len(records)==40,'MATCHED_COMPLETE_B0_TARGET_REPLAY_MEMBERS')
        for row in records:task.need(digest(folder/row['sample_file'])==row['sha256'],'MATCHED_B0_TARGET_OUTPUT_PIN')
    pilot=closed_result(request['native_pilot'])
    summary=pilot['summary'];task.need(summary['complete']and summary['requested']==summary['completed']==2
        and summary['methods']['C500_PLUS_ENDPOINT_BASIS']['task_successes']==2
        and summary['methods']['C500_PLUS_ENDPOINT_BASIS']['hard_safety_stop_conditions']==0
        and summary['methods']['C500_PLUS_ENDPOINT_BASIS']['technical_failures']==0,
        'MATCHED_V3_1_TRUE_NATIVE_SUCCESS_BRANCH')
    return dict(C500_custody=custody,B0_target_replay=request['B0_target_replay'],native_pilot=request['native_pilot'])


def policy_contract(value,checkpoint_files,*,verify_package=True,expected_global_step=2000):
    from .target_gr00t_profile_v1 import PARENT
    task.need(set(value)=={'arm','parent_manifest_sha256','package','Core_matched_qualification'}
        and value['arm']in METHODS and value['parent_manifest_sha256']==PARENT and expected_global_step==2000,
        'MATCHED_EXPLICIT_NEW_POLICY_IDENTITY')
    q=pinned(value['Core_matched_qualification']);family,_=configuration(value['arm'])
    expected=q['C500_package']if family=='C500'else q['SFT2000_package']if family=='SFT2000'else None
    task.need(q['schema']=='CORE_MATCHED128_SMOOTH_ADMISSION_V3_1'and q['admission_class']==PROTOCOL
        and q['preregistration']['sha256']==CONTRACT_SHA and value['package']==expected
        and q['original_cross_hardware_REJECT_preserved']and not q['original_DEV_TEST_admission'],
        'MATCHED_ORIGINAL_PACKAGES_AND_SEPARATE_ADMISSION')
    if verify_package:qualification(q,checkpoint_files)
    return q


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==PROTOCOL
        and panel.get('protocol')==protocol()and panel.get('B1_global_step')==2000
        and panel.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and set(panel.get('policies',{}))==set(METHODS)and len(rows)==128
        and len({r['native_run']for r in rows})==128,'MATCHED_INDEPENDENT_FIXED128')
    for row,(layout,restart,method,seed)in zip(rows,ORDER):
        task.need(row['layout']==layout and row['method']==method and row['parent']==scene_id(layout)
            and row['split']=='TRAIN'and row['purpose']==PURPOSE and row['restart']==restart
            and row['command']==task.COMMAND and row['query_seeds']==list(range(seed,seed+15))
            and Path(row['native_run']).name==row['native_run']and len(row['model_sha256'])==64
            and panel['policies'][method]['arm']==method,'MATCHED_NO_NEW_NOISE_OR_LAYOUT_OR_RETRY')
    return rows


def selection(inp,panel):
    rows=validate_panel(panel);span=inp.get('assignment_slice')
    task.need(inp.get('schema')==INPUT_SCHEMA and inp.get('source_sha')==panel['source_sha']
        and isinstance(span,list)and len(span)==2 and all(type(v)is int for v in span)
        and 0<=span[0]<span[1]<=128,'MATCHED_CANONICAL_UNEXECUTED_CONTIGUOUS_SLICE')
    if span[0]or inp.get('population_predecessor')is not None:
        prior=pinned(inp['population_predecessor'])
        task.need(prior['schema']=='CORE_MATCHED128_POPULATION_ACTUAL_V3_1'
            and prior['panel']==dict(path=inp['panel'],sha256=inp['panel_sha256'])
            and prior['source_sha']==inp['source_sha']and prior['cold_continuation_safe']
            and bool(prior['groups'])and prior['summary']==summarize(panel,prior['outcomes'])
            and [v['native_run']for v in prior['outcomes']]==[r['native_run']for r in rows[:span[0]]],
            'MATCHED_ACTUAL_COLD_PREFIX_NEVER_RETRY_OR_SELECT')
        for group in prior['groups']:
            lease=pinned(group['lease']);result=pinned(group['result'])
            task.need(lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']
                and result['unresolved_pair']is None,'MATCHED_ALL_PRIOR_GROUPS_COLD')
        task.need(all(v['hard_safety_stop_conditions']==v['technical_failures']==0 for v in prior['summary']['methods'].values()),
            'MATCHED_NO_BLIND_CONTINUATION_AFTER_SAFETY_OR_TECHNICAL_FAILURE')
    return rows[span[0]:span[1]]


def summarize(panel,outcomes):
    rows=validate_panel(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)
        and all(r['native_run']in wanted and all(r[k]==wanted[r['native_run']][k]
            for k in('layout','method','restart','parent','split','query_seeds'))for r in outcomes),'MATCHED_NO_FOREIGN_OR_RETRIED_OUTCOME')
    methods={}
    for method in METHODS:
        values=[r for r in outcomes if r['method']==method]
        methods[method]=dict(requested=16,attempted=sum(r['attempted']for r in values),
            task_successes=sum(r['task_success']and r['technical_valid']for r in values),
            valid_task_failures=sum(r['technical_valid']and not r['task_success']for r in values),
            technical_failures=sum(not r['technical_valid']for r in values),
            hard_safety_stop_conditions=sum(r['hard_safety_stop']for r in values),
            actual_GR00T_queries=sum(r.get('actual_GR00T_queries',0)for r in values),
            actual_native_integrations=sum(r.get('actual_native_integrations',0)for r in values))
    return dict(schema='CORE_MATCHED128_SMOOTH_NATIVE_SUMMARY_V3',requested=128,completed=len(outcomes),
        complete=len(outcomes)==128,methods=methods,original_DEV_conditions=0,TEST_conditions=0,
        qualified_training_demonstrations=0,model_updates=0,generalization_claim=False,
        original_cross_hardware_REJECT_replaced=False)


def aggregate(panel_reference,groups):
    """Read closed original groups; retain a contiguous prefix without replay."""
    panel=pinned(panel_reference);rows=validate_panel(panel);outcomes=[];evidence=[];cold=True
    def descriptor(path):return dict(path=str(path),sha256=digest(path))
    for root in map(Path,groups):
        inp=json.loads((root/'INPUT.private.json').read_bytes());lease=json.loads((root/'control/LEASE.json').read_bytes())
        result=json.loads((root/'RESULT.private.json').read_bytes());values=json.loads((root/'OUTCOMES.private.json').read_bytes())
        task.need(inp['panel']==panel_reference['path']and inp['panel_sha256']==panel_reference['sha256']
            and inp['source_sha']==panel['source_sha']and lease['status']=='CLOSED'
            and lease['input_sha256']==digest(root/'INPUT.private.json')and lease['worker_exit_code']==result['exit_code']
            and result['panel_sha256']==panel_reference['sha256']and result['source_sha']==panel['source_sha']
            and result['assignment_slice']==inp['assignment_slice']and inp['assignment_slice'][0]==len(outcomes)
            and result['completed_in_group']==len(values)and result['summary']==summarize(panel,values),
            'MATCHED_CLOSED_GROUP_PROVENANCE_AND_CONTIGUOUS_PREFIX')
        task.need([v['native_run']for v in values]==[r['native_run']for r in rows[len(outcomes):len(outcomes)+len(values)]],
            'MATCHED_EVERY_ATTEMPT_ONCE_IN_FIXED_ORDER')
        if result['exit_code']==0:
            task.need(len(values)==inp['assignment_slice'][1]-inp['assignment_slice'][0]and result['failure']is None,
                'MATCHED_SUCCESSFUL_GROUP_COMPLETES_ITS_SELECTED_SLICE')
        for value in values:
            native=root/'runs'/value['native_run'];spec=json.loads((native/'EXECUTION_SPEC.json').read_bytes())
            for partner in [native,Path(spec['target_gr00t_gpu_root'])]:
                child=json.loads((partner/'control/LEASE.json').read_bytes());worker=json.loads((partner/'WORKER_EXIT.private.json').read_bytes())
                task.need(child['status']=='CLOSED'and child['worker_exit_code']==worker['exit_code']
                    and child['spec_sha256']==digest(partner/'EXECUTION_SPEC.json'),'MATCHED_BOTH_CHILDREN_ACTUALLY_COLD')
            for path,key in [(native/'native-01/RESULT.json','result_sha256'),(Path(spec['target_gr00t_gpu_root'])/'fit-01/RESULT.private.json','GPU_result_sha256')]:
                task.need(value[key]is not None and digest(path)==value[key],'MATCHED_ORIGINAL_RESULT_STILL_IMMUTABLE')
        cold=cold and result['unresolved_pair']is None
        outcomes.extend(values)
        evidence.append(dict(root=str(root),lease=descriptor(root/'control/LEASE.json'),result=descriptor(root/'RESULT.private.json'),outcomes=descriptor(root/'OUTCOMES.private.json')))
    summary=summarize(panel,outcomes)
    return dict(schema='CORE_MATCHED128_POPULATION_ACTUAL_V3_1',panel=panel_reference,source_sha=panel['source_sha'],
        outcomes=outcomes,groups=evidence,summary=summary,cold_continuation_safe=cold,
        planned_conditions=128,attempted_conditions=len(outcomes),unrun_conditions=128-len(outcomes),
        partial_results_select_no_methods=True,original_DEV80_TEST=0)


def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry,scene_id,BASE_SCENE
    from .task_learning_local_policy_v1 import scene_builder_custody
    proof=pinned(spec['task_learning_scene_reference']);layout=proof['layout']
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    builder=scene_builder_custody(panel['registration']['scene_producers'][layout],
        spec['task_learning_scene_reference'],spec['model_path'])
    task.need(layout in LAYOUTS and spec.get('core_local_protocol')==PROTOCOL
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
        and proof['physics_dt_s']==.002,'MATCHED_SAME_SEALED_TRAIN_SCENE_DYNAMICS_AND_SAFETY')
    matches=[r for r in validate_panel(panel)if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'MATCHED_ONE_FIXED_NATIVE_REQUEST');row=matches[0]
    task.need(row['parent']==proof['parent']and row['model_sha256']==digest(spec['model_path'])
        and spec.get('target_gr00t_policy')==panel['policies'][row['method']]
        and spec.get('target_gr00t_query_seeds')==row['query_seeds']
        and spec.get('target_gr00t_pair_id')==row['native_run']
        and spec.get('task_learning_teacher_mode')is None and spec.get('task_learning_control')is None,
        'MATCHED_EXACT_C500_METHOD_AND_NO_TEACHER_FALLBACK')
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
        and document_hash(s['checkpoint_files'])==CHECKPOINT_INVENTORY,'MATCHED_ORIGINAL_NUMERICS_AND_MODEL_INVENTORY')
    descriptor=s['producer_native_spec'];producer=pinned(descriptor);task.declared(producer)
    task.need(Path(descriptor['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('task_learning_role')==PROTOCOL and producer.get('core_local_protocol')==PROTOCOL
        and producer.get('target_gr00t_gpu_root')==s['registered_root']and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and producer.get('target_gr00t_pair_id')==s.get('target_gr00t_pair_id')
        and producer.get('target_gr00t_B1_global_step')==s.get('target_gr00t_B1_global_step')==2000,
        'MATCHED_EXACT_CURRENT_NATIVE_AND_NEURAL_PAIR')
    native_scene_admission(producer)
    q=policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model)
    task.need(s.get('GPU_uuid')==q['GPU_uuid'],'MATCHED_SAME_QUALIFIED_CORE_CARD')
    memory_contract(s);return producer


def load(s,root):
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen
    from .target_gr00t_core_v1 import trim_unused_logits
    from .target_gr00t_delivery_v1 import install
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    value=s['target_gr00t_policy'];q=policy_contract(value,s['checkpoint_files'],verify_package=False)
    family,basis=configuration(value['arm']);processor_package=q['SFT2000_package']
    if family=='C500':
        from .task_learning_endpoint_native_v3 import load as qualified_C500_load
        instance,torch=qualified_C500_load(s,root)
    else:
        initialize=None
        if family=='SFT2000':
            def initialize(model):
                install(model,processor_package['path'],processor_package['sha256'],s['checkpoint_files'])
                return dict(package=processor_package,ordinary_CPU_install=True)
        instance,torch=_load_frozen(s,root,cpu_model_initializer=initialize)
        trim_unused_logits(instance.model)
    if family in('B0_TARGET_PROCESSOR','SFT2000'):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
        instance.processor=Gr00tN1d7Processor.from_pretrained(Path(processor_package['path'])/'processor',
            model_name=local_model_name(s['registered_root'],s['neural_bindings']),local_files_only=True,
            transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
                cache_dir=s['neural_bindings']['cosmos_hub'],revision=COSMOS_REVISION))
        instance.processor.eval()
    instance.identity=dict(policy=value,weight_family=family,Processor='ORIGINAL_B0'if family=='B0'else processor_package,
        numeric_mode=s['numeric_mode'],representation='SAME_CURRENT_STATE_PREDICTED_GOAL_QUINTIC'if basis else'RAW')
    write_json(Path(root)/'MATCHED_POLICY_LOADED.private.json',dict(identity=instance.identity,actual_model_loads=1,
        ordinary_internal_GR00T_weights=True,optimizer_updates=0,external_inference_adapter=False))
    return instance,torch


def register_panel(request):
    """Use the existing owner/Journal/registry; seal128 new diagnostic requests."""
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
    source=Path(request['source_root']);root=Path(request['root']);q=pinned(request['Core_matched_qualification'])
    task.need(root.parent==Path(n['registered_root']).parents[2]and source.resolve()==source,
              'LOCAL_PANEL_UNDER_ORIGINAL_PROJECT_DATA_ROOT')
    task.need(all(digest(source/name)==pin for name,pin in request['source_files'].items()),
              'SEALED_LOCAL_RUNTIME_SOURCE')
    from .target_gr00t_profile_v1 import PARENT
    policies={m:dict(arm=m,parent_manifest_sha256=PARENT,package=(q['C500_package']if configuration(m)[0]=='C500'else q['SFT2000_package']if configuration(m)[0]=='SFT2000'else None),
                    Core_matched_qualification=request['Core_matched_qualification'])for m in METHODS}
    for m in policies:policy_contract(policies[m],g['checkpoint_files'])
    rows=[]
    for index,(layout,restart,method,seed)in enumerate(ORDER):
        descriptor=request['scenes'][layout];proof=pinned(descriptor)
        model=Path(descriptor['path']).parent/'SCENE.private.mjb'
        scene_builder_custody(request['scene_producers'][layout],descriptor,model)
        task.need(proof['layout']==layout and proof['split']=='TRAIN'
            and proof['compiled']['scene_sha256']==digest(model),'PREREGISTERED_TRAIN_SCENE_NOT_DEV_TEST')
        generation=root.name.rsplit('-',1)[-1]
        task.need(generation.isdigit(),'UNIQUE_LOCAL_REPAIRED_PANEL_GENERATION')
        rows.append(dict(native_run=f'tl-core-matched-20261111-v{generation}-{index:02d}-{method.lower()}',method=method,
            layout=layout,parent=scene_id(layout),restart=restart,split='TRAIN',purpose=PURPOSE,
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
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',planned_conditions=128,
                panel_sha256=digest(root/'PANEL.private.json'),created_epoch=time.time()))
            reg,pin=snapshot(n['registry'])
            task.need(not any(e.get('path')==str(root)for e in reg['entries']),'ONE_LOCAL_PANEL_REGISTRY')
            reg['entries'].append(dict(path=str(root),role=key,scope=SCHEMA,owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],reg,key+'-REGISTRY',expected_before_sha256=pin)
            return dict(status='CORE_MATCHED128_SMOOTH_128_CONDITIONS_SEALED',panel=dict(path=str(root/'PANEL.private.json'),sha256=digest(root/'PANEL.private.json')))
        return journal.once(key,dict(source_sha=request['source_sha'],qualification=request['Core_matched_qualification']),effect)
