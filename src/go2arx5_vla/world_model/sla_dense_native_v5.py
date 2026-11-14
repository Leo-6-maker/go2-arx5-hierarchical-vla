"""Distinct complete3K dense policy admission and fixed raw-H40/P10 TRAIN8.

This reuses the registered native/neural executor. It does not admit a live
training snapshot, a four-update qualification, old DEV/TEST or teacher data.
"""
import json
from pathlib import Path

from . import task_learning_grasp_v1 as task
from .storage import digest
from . import sla_reference_native_v6 as reference
from .stable_task_scene_v1 import pinned

PROTOCOL = task.SLA_DENSE_ROLE
SCHEMA = 'SLA_DENSE3K_RAW_P10_TRAIN8_PANEL_V5'
INPUT_SCHEMA = 'SLA_DENSE3K_RAW_P10_TRAIN8_GROUP_INPUT_V5'
NATIVE_SCHEMA = 'SLA_DENSE3K_RAW_P10_NATIVE_ACTUAL_V5'
EXPERIMENT = 'LIMITED_TWO_EPISODE_DENSE3K_RAW_P10_TRAIN8_NOT_DEV_TEST'
ADMISSION = 'SLA_COMPLETE_DENSE3K_CORE_RGB16_NATIVE_ADMISSION_V5'
METHODS = ('DENSE_FM3K_RAW_P10','DENSE_SF3K_RAW_P10')
NAMED_METHOD = 'N20_FM3K_RAW_P10'
NAMED_ADMISSION = 'SLA_FRESH_N20_FM3K_CORE_RGB16_NATIVE_ADMISSION_V5'
NAMED_SCHEMA = 'SLA_FRESH_N20_RAW_P10_TRAIN8_PANEL_V5'
NAMED_INPUT_SCHEMA = 'SLA_FRESH_N20_RAW_P10_TRAIN8_GROUP_INPUT_V5'
NAMED_DECLARATION = 'experiments/sla_go2_posttrain/named20_train8_inputs_v5.json'
NAMED_EXPERIMENT = 'FRESH_N20_COMPLETE3K_RAW_P10_TRAIN8_NOT_INDEPENDENT_DEV_OR_D100'
NAMED_DATASET = 'T_PICK_HOLD_TRAIN8_P10_N20_V5'
NAMED_RECIPE = 'SLA_FRESH_NAMED_N20_FM32_3K_V5'
QUERIES, PREFIX_ATOMS, FINAL_HOLD = 60, 10, 40
ENDPOINT = task.B + 25*(QUERIES*PREFIX_ATOMS+FINAL_HOLD)
DECLARATION = 'experiments/sla_go2_posttrain/dense_train8_inputs_v5.json'
RECIPES = dict(zip(METHODS,('SLA_NAMED_DENSE_PICK_HOLD_FM32_3K_V4_1',
                          'SLA_NAMED_DENSE_PICK_HOLD_SF32_3K_ALPHA_HALF_V5')))


def parameter_bytes(inventory):
    import math
    sizes={'bfloat16':2,'float16':2,'float32':4,'float64':8}
    rows=inventory['parameters']
    task.need(bool(rows) and all(row['dtype'] in sizes and row['numel']==math.prod(row['shape'])for row in rows),
              'DENSE_ACTUAL_PARAMETER_INVENTORY_DTYPES_AND_SHAPES')
    return sum(row['numel']*sizes[row['dtype']]for row in rows)


def protocol():
    from .sla_native_v1 import protocol as original
    from .sla_dense_policy_v5 import VERSION
    return dict(original(),stage=PROTOCOL,scene_family='FIXED_DENSE3K_RAW_P10_TARGET_EVALUATION',
        policy='SLA_COMPLETE_DENSE3K_RAW_H40_WITHOUT_ENDPOINT_BASIS',
        prefix_atoms=PREFIX_ATOMS,policy_queries=QUERIES,final_hold_atoms=FINAL_HOLD,
        consumer_version=VERSION,reference_TTL_s=.4,reference_renewal_max_s=.35,
        complete_H40_gate_before_every_prefix=True,unexecuted_suffix_is_expert_data=False)


def declaration(*,named=False,corrected=False):
    p=Path(__file__).resolve().parents[3]/(reference.DECLARATION if corrected else NAMED_DECLARATION if named else DECLARATION)
    return json.loads(p.read_bytes()),digest(p)


def qualification(q,*,verify_package=True):
    """Check actual closed training/save and fresh ordinary-RGB reconstruction."""
    from .sla_native_v1 import closed
    from ..sla_adapter.gripper_latent_v2 import VERSION, EPSILON
    named=q.get('schema')==NAMED_ADMISSION
    corrected=q.get('schema')==reference.ADMISSION
    task.need((q.get('schema')==ADMISSION and q.get('method') in METHODS
               or named and q.get('method')==NAMED_METHOD
               or corrected and q.get('method')==reference.METHOD)
              and q.get('old_DEV_TEST_admitted') is False,'DENSE_COMPLETE3K_SEPARATE_IDENTITY')
    owner=closed(q['candidate_owner'])
    rgb_owner=closed(q['ordinary_RGB_owner'])
    training=pinned(q['training_result']);rgb=pinned(q['ordinary_RGB_reconstruction'])
    for proof,owner_pin in ((q['training_result'],q['candidate_owner']),
                            (q['parameter_inventory'],q['candidate_owner']),
                            (q['ordinary_RGB_reconstruction'],q['ordinary_RGB_owner'])):
        task.need(Path(proof['path']).resolve().is_relative_to(Path(owner_pin['path']).parent.resolve()),
                  'DENSE_RESULT_INSIDE_ITS_CLOSED_EXECUTION_OWNER')
    task.need(q['ordinary_parameter_bytes']==parameter_bytes(pinned(q['parameter_inventory'])),
              'DENSE_REUSED_ACTUAL_MODEL_PARAMETER_BYTES')
    task.need(training.get('step')==3000 and training.get('cursor')==96000
        and training.get('exit_code')==0 and training.get('physical_requests')==0
        and training['actual']['optimizer_updates']==3000
        and training['actual']['completed_training_visits']>=96000,
        'DENSE_REAL_COMPLETE3K_AND96000_COMMITTED_TARGET_VISITS')
    if corrected:
        reference.endpoint(training,owner)
    elif named:
        task.need(training['schema']=='SLA_FRESH_N20_FINITE_FM3K_ACTUAL_V5'
            and training['operation']=='NAMED_PICK_HOLD_N20_FM_3K'
            and not training['noncandidate'] and training['complete3K'] is True
            and training['fresh_complete_episodes']==20 and training['old_2EP_reused']==0
            and training['D100_new'] is training['D300_new'] is False
            and owner['schema']=='SLA_CORE_FRESH_N20_FM3K_RGB_ACTUAL_V5'
            and owner['complete3K'] is owner['passes'] is owner['fresh_RGB16_bit_exact'] is True
            and owner['step']==3000 and owner['target_visits']==96000 and owner['source_visits']==0
            and owner['fresh_complete_episodes']==20,
            'NAMED_N20_ACTUAL_FRESH20_FM3K_NOT_OLD2EP_OR_QUALIFICATION')
    elif q['method']==METHODS[1]:
        task.need(training['schema']=='SLA_CLEAN_BASE_DENSE_SPATIAL3K_ACTUAL_V5'
            and training['source_visits']==0 and training['actual']['teacher_model_loads']==0
            and training['actual']['spatial_cache_training_visits']>=96000
            and owner.get('complete3K') is True and owner.get('passes') is True
            and owner['candidate_updates']==3000 and owner['source_visits']==0,
            'DENSE_ACTUAL_TARGET_ONLY_SPATIAL_ENDPOINT')
    else:
        task.need(training['schema']=='SLA_DENSE_PILOT_FINITE_CUDA_ACTUAL_V4_1'
            and training['operation']=='DENSE_PICK_HOLD_3K' and not training['noncandidate'],
            'DENSE_ACTUAL_FM3K_NOT_CORE4_QUALIFICATION')
    complete=pinned(q['complete_state'])
    task.need(complete.get('schema')=='SLA_COMPLETE_PEFT_STATE_V1'
        and complete['step']==3000 and complete['cursor']==96000
        and len(complete['arrays'])==(105 if q['method']==METHODS[1] else 87)
        and q['complete_state']['sha256']==training['last_checkpoint']['sha256'],
        'DENSE_COMPLETE_PEFT_ADAM_RNG_AND_SPATIAL_PROJECTOR_STATE')
    m=pinned(dict(path=str(Path(q['package']['path'])/'MANIFEST.private.json'),sha256=q['package']['sha256']))
    b=m['bindings']
    task.need(m.get('schema')=='SLA_GRIP_LOGIT_ORDINARY_RGB_INFERENCE_PACKAGE_V3'
        and m.get('codec')==b.get('codec_version')==VERSION and m.get('label_epsilon')==EPSILON
        and m.get('step')==3000 and m.get('cursor')==96000 and not m['qualification_only']
        and m['original_complete_sha256']==q['complete_state']['sha256']
        and b==complete['bindings'] and b['recipe']==(reference.RECIPE if corrected else NAMED_RECIPE if named else RECIPES[q['method']])
        and b['dataset_id']==(reference.DATASET_ID if corrected else NAMED_DATASET if named else 'T_PICK_HOLD_TRAIN01_P10_2EP_V4_1')
        and b['dense_manifest_sha256']==q['dense_manifest_sha256']
        and b['derived_statistics_sha256']==m['statistics_sha256']
        and b['D100_new'] is False and b['PICK_PLACE_admitted'] is False
        and m['parent_manifest']['sha256']==q['parent_manifest']['sha256']
        and digest(q['parent_manifest']['path'])==m['parent_manifest']['sha256']
        and Path(q['parent_params']).resolve()==(Path(q['parent_manifest']['path']).parent/'model/params').resolve()
        and m['image_size']==[224,224] and m['image_masks']==[True,True,False]
        and m['action_horizon']==40 and m['label_interval_s']==.05 and m['sampler_steps']==10
        and all(m.get(k) is False for k in ('endpoint_basis_applied','state_gripper_transformed',
                    'teacher_required','training_cache_required','optimizer_required'))
        and m['decode_order']=='INVERSE_LATENT_NORMALIZATION_THEN_SIGMOID_ONCE_THEN_ORIGINAL_NATIVE_NOMINAL_MAPPING',
        'DENSE_EXACT_TRAINED_RAW_LOGIT_RGB_PACKAGE_WITHOUT_ENDPOINT_BASIS')
    audit=pinned(q['target_audit'])
    task.need(q['target_audit']['sha256']==b['target_audit'],'DENSE_SAME_EXACT_DATA_AUDIT_BYTES')
    if corrected:
        task.need(b['reference_conversion_audit']==q['target_audit']['sha256'],
                  'REFERENCE_SAME_CONVERSION_BYTES_IN_COMPLETE_AND_PACKAGE')
        reference.labels(audit,b,q['dense_manifest_sha256'])
        task.need(owner['dataset_manifest_sha256']==b['dense_manifest_sha256'],
                  'REFERENCE_SAME_CLOSED_OWNER_DATASET')
    else:
        task.need(audit['passes'] and audit['read_windows']==(1220 if named else 122)
            and audit['manifest_sha256']==b['dense_manifest_sha256'],'DENSE_REUSED_EXACT122_WINDOW_DATA_AUDIT')
    if named:
        task.need(audit['schema']=='SLA_FRESH_NAMED_N20_ALL_WINDOW_READBACK_V5'
            and audit['dataset_id']==NAMED_DATASET and audit['episodes']==20
            and b['old_2EP_reused']==b['old_engineering16_reused']==0 and b['D300_new'] is False
            and b['limited_fresh_named_N20_fitting'] is True,
            'NAMED_N20_TYPED_ALL1220_WINDOW_READER_AND_UNMODIFIED_OLD_DATA')
    task.need(rgb.get('passes') is True and rgb.get('all_bit_exact') is True
        and rgb['ordinary_RGB_generations']==len(rgb['records'])==16
        and rgb['actual_parent_loads']==1 and rgb['optimizer_updates']==rgb['physical_requests']==0
        and rgb['teacher_loaded'] is False and rgb['optimizer_loaded'] is False,
        'DENSE_ACTUAL_FRESH_RGB16_WITHOUT_TEACHER_OR_OPTIMIZER')
    rs=pinned(q['ordinary_RGB_spec'])
    task.need(Path(q['ordinary_RGB_spec']['path']).parent==Path(q['ordinary_RGB_reconstruction']['path']).parent
        and rs['package']==q['package'] and rs['GPU_uuid']==q['GPU_uuid']
        and rs['parent_params']==q['parent_params'] and rs['python']==q['runtime']['python']
        and rs['upstream_root']==q['runtime']['upstream_root']
        and rs['neural_owner_lock']==q['neural_owner_lock']
        and rs['JAX_memory_fraction']==q['runtime']['JAX_memory_fraction']
        and rs['allocator_limit_bytes']==q['allocator_limit_bytes']
        and max(rgb['GPU_memory']['peak_bytes_in_use'],rgb['GPU_memory']['bytes_limit'])<=q['allocator_limit_bytes'],
        'DENSE_SAME_ACTUAL_CORE_RGB_DEVICE_ENVIRONMENT_AND_CAPACITY')
    if verify_package:
        root=Path(q['package']['path'])
        for row in m['adapter_arrays']:
            p=root/row['file']
            task.need(p.resolve().is_relative_to(root.resolve()) and digest(p)==row['sha256'],
                      'DENSE_ALL_DEPLOYED_ADAPTER_BYTES')
        for name,key in (('STATISTICS.json','statistics_sha256'),('paligemma_tokenizer.model','tokenizer_sha256')):
            task.need(digest(root/name)==m[key],'DENSE_EXACT_STATISTICS_AND_TOKENIZER')
    return q


def policy_contract(value,checkpoint_files=None,*,verify_package=True,expected_global_step=3000):
    task.need(set(value)=={'arm','package','Native_body_qualification'}
        and value['arm'] in METHODS+(NAMED_METHOD,reference.METHOD) and expected_global_step==3000,'DENSE_ONLY_FIXED3K_POLICY')
    q=qualification(pinned(value['Native_body_qualification']),verify_package=verify_package)
    task.need(q['method']==value['arm'] and q['package']==value['package'],'DENSE_SAME_METHOD_AND_PACKAGE')
    return q


def memory_contract(spec):
    q=policy_contract(spec['target_gr00t_policy'],verify_package=False)
    task.need(spec['memory_execution']=='SLA_ORDINARY_RGB_JAX_SINGLE_CORE_V1'
        and spec['memory_execution_reference']==q['ordinary_RGB_reconstruction']
        and spec['allocator_limit_bytes']==q['allocator_limit_bytes'],'DENSE_QUALIFIED_RGB_MEMORY')
    return q


def validate_panel(panel):
    from . import sla_dev48_v5 as dev
    if panel.get('schema')==dev.SCHEMA:return dev.validate_panel(panel)
    named=panel.get('schema')==NAMED_SCHEMA
    corrected=panel.get('schema')==reference.SCHEMA
    d,pin=declaration(named=named,corrected=corrected);rows=panel.get('assignments')or[];method=panel.get('method')
    task.need(panel.get('schema')==(reference.SCHEMA if corrected else NAMED_SCHEMA if named else SCHEMA) and panel.get('task_learning_role')==PROTOCOL
        and panel['protocol']==protocol() and panel['B1_global_step']==3000
        and panel['population']=='TRAIN8' and method in ((reference.METHOD,)if corrected else (NAMED_METHOD,)if named else METHODS)
        and set(panel['policies'])=={method} and panel['policies'][method]['arm']==method
        and panel['preregistration']['sha256']==pin and pinned(panel['preregistration'])==d
        and len(rows)==len({r['native_run']for r in rows})==8,'DENSE_FIXED_TRAIN8_POPULATION_BEFORE_OUTPUTS')
    for row,expected in zip(rows,d['conditions'],strict=True):
        task.need(all(row[k]==expected[k]for k in ('layout','restart','query_seeds'))
            and row['method']==method and row['split']=='TRAIN'
            and row['parent']=='TASK_LEARNING_UNDER_HAND_'+row['layout']
            and row['command']==task.COMMAND and row['purpose']==(reference.EXPERIMENT if corrected else NAMED_EXPERIMENT if named else EXPERIMENT)
            and len(row['model_sha256'])==64 and Path(row['native_run']).name==row['native_run'],
            'DENSE_ALL_EIGHT_CONDITIONS_AND_SIXTY_DECLARED_QUERY_SEEDS')
    return rows


def selection(inp,panel):
    from . import sla_dev48_v5 as dev
    if panel.get('schema')==dev.SCHEMA:return dev.selection(inp,panel)
    rows=validate_panel(panel)
    if panel.get('named_cold_suffix_v5')is not None:
        from .sla_n20_suffix_v5 import selection as suffix_selection
        return suffix_selection(inp,panel,rows)
    task.need(inp.get('schema')==(reference.INPUT_SCHEMA if panel['schema']==reference.SCHEMA else NAMED_INPUT_SCHEMA if panel['schema']==NAMED_SCHEMA else INPUT_SCHEMA) and inp['source_sha']==panel['source_sha']
        and inp['assignment_slice']==[0,8] and inp.get('population_predecessor') is None,
        'DENSE_ORIGINAL_FULL_TRAIN8_NOT_SELECTED_SUCCESSES')
    return rows


def native_scene_admission(spec):
    from .sla_native_v1 import scene_custody
    from .task_learning_scene_v1 import validate_static_provenance
    from .native_camera_transfer_v1 import CAMERA_ID
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    proof=pinned(spec['task_learning_scene_reference'])
    task.need(spec['task_learning_role']==spec['core_local_protocol']==PROTOCOL,'DENSE_DISTINCT_NATIVE_ROLE')
    builder=scene_custody(panel['registration']['scene_producers'][proof['layout']],
                          spec['task_learning_scene_reference'],spec['model_path'])
    validate_static_provenance(spec,proof,expected_role=PROTOCOL,verified_builder_sha256=builder)
    matches=[r for r in validate_panel(panel)if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1,'DENSE_ONE_REGISTERED_NATIVE_REQUEST')
    row=matches[0]
    task.need(row['model_sha256']==digest(spec['model_path'])
        and spec['target_gr00t_policy']==panel['policies'][row['method']]
        and spec['target_gr00t_query_seeds']==row['query_seeds']
        and spec['target_gr00t_B1_global_step']==3000
        and spec['target_gr00t_pair_id']==row['native_run']
        and spec['observation_camera_profile']==CAMERA_ID
        and spec.get('task_learning_teacher_mode') is None and spec.get('task_learning_control') is None
        and not spec.get('expert_eef_stream_v2'),'DENSE_CURRENT_BODY_WRIST_RAW_POLICY_WITHOUT_TEACHER')
    return proof


def validate_live(spec,*,verify_model=True):
    from .task_learning_live_pair_v1 import PROFILE,KIND
    from .sla_r1_diagnostic_v5 import inference_identity
    task.need(spec['core_local_protocol']==PROTOCOL and spec['profile']==PROFILE and spec['kind']==KIND
        and spec['execution_site']=='CORE_FUSION_NEURAL' and spec['physics'] is False
        and spec['device']=='cuda' and spec['max_queries']==QUERIES
        and spec['GR00T_updates']==spec['WM_updates']==spec['Actor_updates']==0
        and spec['numeric_mode']=='SLA_PINNED_JAX_BF16_DETERMINISTIC_V1'
        and spec['baseline_changed'] is False and spec['task_learning_grasp']==protocol(),
        'DENSE_SIXTY_QUERIES_SINGLE_RGB_MODEL_ONLY')
    producer=pinned(spec['producer_native_spec']);task.declared(producer)
    task.need(Path(spec['producer_native_spec']['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer['source_sha']==spec['source_sha'] and producer['target_gr00t_gpu_root']==spec['registered_root']
        and producer['task_learning_role']==producer['core_local_protocol']==PROTOCOL
        and all(producer[k]==spec[k]for k in ('target_gr00t_policy','target_gr00t_query_seeds',
                  'target_gr00t_pair_id','target_gr00t_B1_global_step','observation_camera_profile')),
        'DENSE_EXACT_NATIVE_NEURAL_PAIR')
    native_scene_admission(producer)
    q=policy_contract(spec['target_gr00t_policy'],verify_package=verify_model)
    rs=pinned(q['ordinary_RGB_spec'])
    inference_identity(spec['source_root'],rs['source_root'],logit=True)
    task.need(spec['GPU_uuid']==q['GPU_uuid'],'DENSE_ACTUALLY_RECONSTRUCTED_GPU')
    memory_contract(spec)
    return producer


def summarize(panel,outcomes):
    rows=validate_panel(panel)
    task.need(len(outcomes)<=len(rows) and all(all(r[k]==rows[i][k]for k in
        ('native_run','layout','method','restart','parent','split','query_seeds'))
        for i,r in enumerate(outcomes)),'DENSE_ORIGINAL_CONDITIONS_ONCE')
    return dict(schema=panel['schema']+'_ACTUAL',population=panel['population'],requested=len(rows),completed=len(outcomes),complete=len(outcomes)==len(rows),
        methods={panel['method']:dict(requested=len(rows),attempted=sum(r['attempted']for r in outcomes),
            task_successes=sum(r['technical_valid']and r['task_success']for r in outcomes),
            valid_task_failures=sum(r['technical_valid']and not r['task_success']for r in outcomes),
            technical_failures=sum(not r['technical_valid']for r in outcomes),
            hard_safety_stop_conditions=sum(r['hard_safety_stop']for r in outcomes),
            actual_SLA_queries=sum(r.get('actual_SLA_queries',0)for r in outcomes),
            actual_native_integrations=sum(r.get('actual_native_integrations',0)for r in outcomes))},
        original_DEV_conditions=0,TEST_conditions=0,model_updates=0,
        qualified_training_demonstrations=0,WBC_changes=False,generalization_claim=False)


def register_panel(request):
    """Seal one method's full fixed population via the existing registry/Journal."""
    from contextlib import closing
    import time
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .task_learning_student_registration_v2 import closed_base,actual_site,descriptor
    from .sla_native_v1 import scene_custody
    from . import sla_dev48_v5 as dev
    n=closed_base(request['native_base']);closed_base(request['gpu_base'])
    campaign=Path(n['campaign_root']);actual_site(campaign)
    root,source=Path(request['root']),Path(request['source_root'])
    task.need(root.parent==Path(n['registered_root']).parents[2] and source.resolve()==source,
              'DENSE_PANEL_INSIDE_EXISTING_PROJECT')
    seal=pinned(request['source_seal'])
    task.need(seal['source_sha']==request['source_sha'] and seal['files']==request['source_files']
        and Path(request['source_seal']['path'])==source/'SOURCE_STAGED.private.json',
        'DENSE_EXISTING_IMMUTABLE_SOURCE_TRANSFER_SEAL')
    q=qualification(pinned(request['Native_body_qualification']))
    method=q['method'];policy=dict(arm=method,package=q['package'],Native_body_qualification=request['Native_body_qualification'])
    named=q['method']==NAMED_METHOD
    corrected=q['method']==reference.METHOD
    development=request.get('population')==dev.POPULATION
    task.need(request.get('population','TRAIN8')in('TRAIN8',dev.POPULATION),'DENSE_EXPLICIT_TRAIN8_OR_NEW_DEV48')
    d,pin=dev.declaration()if development else declaration(named=named,corrected=corrected)
    inputs=dev.inputs()if development else d['conditions']
    purpose=dev.EXPERIMENT if development else reference.EXPERIMENT if corrected else NAMED_EXPERIMENT if named else EXPERIMENT
    task.need(request['preregistration']['sha256']==pin and pinned(request['preregistration'])==d
        and set(request['scenes'])=={r['layout']for r in inputs},'DENSE_FIXED_INPUTS_AND_SCENES')
    if development:dev.disjointness(request['scene_disjointness'])
    rows=[]
    for expected in inputs:
        layout=expected['layout'];reference=request['scenes'][layout];proof=pinned(reference)
        model=Path(reference['path']).parent/'SCENE.private.mjb'
        scene_custody(request['scene_producers'][layout],reference,model)
        rows.append(dict(expected,native_run=f"sla-{method.lower()}-{layout.lower()}-r{expected['restart']}",
            parent=proof['parent'],method=method,model_sha256=digest(model),split=dev.SPLIT if development else'TRAIN',
            purpose=purpose,command=task.COMMAND))
    panel=dict(schema=dev.SCHEMA if development else reference.SCHEMA if corrected else NAMED_SCHEMA if named else SCHEMA,task_learning_role=PROTOCOL,protocol=protocol(),source_sha=request['source_sha'],
        B1_global_step=3000,training_seed=20261112,method=method,population=dev.POPULATION if development else'TRAIN8',policies={method:policy},
        assignments=rows,preregistration=request['preregistration'],
        registration={k:request[k]for k in ('source_root','source_files','native_base','gpu_base','scenes',
            'scene_producers','scene_allowlist','scene_allowlist_sha256','memory_execution','memory_execution_reference')})
    if development:panel['scene_disjointness']=request['scene_disjointness']
    if request.get('named_cold_suffix_v5')is not None:
        task.need(named and not development,'N20_ONLY_EXPLICIT_TRAIN8_COLD_SUFFIX')
        panel['named_cold_suffix_v5']=request['named_cold_suffix_v5']
        panel['registration']['native_pause_read_wait']=request['native_pause_read_wait']
        from .sla_n20_suffix_v5 import prefix
        prefix(panel)
    if request.get('capacity_wait_before_pair_v3')is not None:
        from .sla_native_resource_continuation_v3 import configuration
        panel['registration']['capacity_wait_before_pair_v3']=request['capacity_wait_before_pair_v3'];configuration(panel)
    if corrected:
        from .sla_native_continuation_v2 import wait_seconds
        task.need(wait_seconds(dict(task_learning_role=PROTOCOL,native_pause_read_wait=request['native_pause_read_wait']))==90,
                  'REFERENCE_REUSE_ORIGINAL_EXACT_ACK_READ_ALLOWANCE')
        panel['registration']['native_pause_read_wait']=request['native_pause_read_wait']
    validate_panel(panel);key=root.name.upper()+'-PANEL';journal=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists() and not journal.path(key).exists(),'DENSE_UNKNOWN_PANEL_NEVER_REPLAY')
        def effect():
            root.mkdir();(root/'control').mkdir();write(root/'PANEL.private.json',panel)
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',planned_conditions=len(rows),
                panel_sha256=digest(root/'PANEL.private.json'),created_epoch=time.time()))
            registry,before=snapshot(n['registry'])
            task.need(not any(r.get('path')==str(root)for r in registry['entries']),'DENSE_NEW_PANEL_REGISTRY')
            registry['entries'].append(dict(path=str(root),role=key,scope=panel['schema'],owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],registry,key+'-REGISTRY',expected_before_sha256=before)
            return dict(status='DENSE_FIXED_PANEL_SEALED_NO_WORKER_STARTED',panel=descriptor(root/'PANEL.private.json'))
        return journal.once(key,dict(policy=policy,preregistration=request['preregistration'],source_sha=request['source_sha']),effect)
