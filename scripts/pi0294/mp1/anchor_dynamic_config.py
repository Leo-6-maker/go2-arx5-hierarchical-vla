"""N1's finite saved-state dynamics gate, not a new live policy rollout."""
import os
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import n1_limits as n1

SCHEMA='MP1_N1_EIGHT_ANCHOR_DYNAMIC_PARITY_V1'
SCOPE='MP1_N1_SAVED_ANCHOR_DYNAMIC_PARITY_NO_LIVE_OR_NEURAL'
CHILD='n1-anchor-dynamic-parity-01'
WALL_S=3600
CAPS=dict(main_starts=8,recovery_forks=0,live_and_recovery_integrations=0,
    shadow_integrations=1600,generation_queries=0,optimizer_updates=0)
CPU_CAPS=dict(core=3600,hopper=0)
CARD='plans/N1_ANCHOR_DYNAMIC_PARITY_V1.json'
INPUTS='plans/ANCHOR_DYNAMIC_INPUTS.private.json'
TRACE_SCHEMA='MP1_N1_SINGLE_FAILED_ANCHOR_TRACE_DIAGNOSTIC_V1'
TRACE_CHILD='n1-anchor-failure-trace-02'
TRACE_CARD='plans/N1_ANCHOR_FAILURE_TRACE_V1.json'
ABLATION_SCHEMA='MP1_N1_FAILED_ANCHOR_NATIVE_COPY_ABLATION_V1'
ABLATION_CHILD='n1-anchor-native-copy-ablation-03'
ABLATION_CARD='plans/N1_ANCHOR_NATIVE_COPY_ABLATION_V1.json'
TRACE02_FAILURE='6054695caaac588df42a36529a7793c83d63c2d375140aca991b3abd5cbd7294'
NATIVE_GATE_SCHEMA='MP1_N1_EIGHT_NATIVE_HISTORY_COPY_PARITY_V1'
NATIVE_GATE_CHILD='n1-anchor-native-history-gate-04'
NATIVE_GATE_CARD='plans/N1_EIGHT_NATIVE_HISTORY_COPY_PARITY_V1.json'
COPY_MECHANISM_RESULT='b18b1e47bb059c3153a2a69697f8e1a704a6ea0915ebf8e6c10487e3c797af6c'
ENDPOINT_SCHEMA='MP1_N1_ENDPOINT_NI16_MATCHED_SHADOWS_V1'
ENDPOINT_CHILD='n1-endpoint-ni16-matched-shadows-05'
ENDPOINT_CARD='plans/N1_ENDPOINT_NI16_MATCHED_SHADOWS_V1.json'
FAILED_STATE='PI0294-MP1-S-BC-512-TRAIN4-F1-000-T1-S026'
PRIOR_FAILURE='03a229d3e8ac414bcb54c42408ad7d83fcb892a073ecd9e3c0c43de15841e6c6'
PRIOR_LEDGER='d4926aa3d22c961d348b7129e5016f72307c6b7d550c012af7cfe754c66c5ca5'
B1_RESULT='c8dc20f5eda85c33f36986b5dbc79bcfdf375f46991a5ed7965429d4d4fdb7d9'
B1_CLOSED='d1b4f818d7cb8724d9c1e2debf35e22b6ffa55373d3f99c2438538668e7301d8'
B1_MANIFEST='42b96f244d1d51c37950b3f12d014dfaf53f663cb8a00a4492021e24b4c3ff76'
B1_SOURCE_INPUTS='99549abd8cb8e6479862e6a07e3fb5574d53b1be3f7296758ad366541971082b'


def is_diagnostic(stage):return stage.get('schema') in [SCHEMA,TRACE_SCHEMA,ABLATION_SCHEMA,NATIVE_GATE_SCHEMA,ENDPOINT_SCHEMA]


def specification(stage):
    if stage.get('schema')==ENDPOINT_SCHEMA:
        return dict(child=ENDPOINT_CHILD,scope='MP1_N1_ORIGINAL_EIGHT_NI16_MATCHED_SHADOWS_NO_LIVE',
            caps=dict(CAPS,main_starts=4,shadow_integrations=32800),CPU_core_s=dict(core=3600,hopper=0),
            wall_s=3600,card=ENDPOINT_CARD,states=8,parents=4,trace_only=False,native_gate_only=True,endpoint_only=True)
    if stage.get('schema')==SCHEMA:
        return dict(child=CHILD,scope=SCOPE,caps=CAPS,CPU_core_s=CPU_CAPS,wall_s=WALL_S,
            card=CARD,states=8,parents=4,trace_only=False)
    if stage.get('schema')==TRACE_SCHEMA:
        return dict(child=TRACE_CHILD,scope='MP1_N1_SINGLE_FAILED_ANCHOR_TRACE_NOT_FULL_QUALIFICATION',
            caps=dict(CAPS,main_starts=1,shadow_integrations=200),CPU_core_s=dict(core=600,hopper=0),
            wall_s=600,card=TRACE_CARD,states=1,parents=1,trace_only=True)
    if stage.get('schema')==ABLATION_SCHEMA:
        return dict(child=ABLATION_CHILD,scope='MP1_N1_SINGLE_FAILED_ANCHOR_NATIVE_COPY_ABLATION',
            caps=dict(CAPS,main_starts=2,shadow_integrations=5600),CPU_core_s=dict(core=1800,hopper=0),
            wall_s=1800,card=ABLATION_CARD,states=1,parents=1,trace_only=True,ablation_only=True)
    if stage.get('schema')==NATIVE_GATE_SCHEMA:
        return dict(child=NATIVE_GATE_CHILD,scope='MP1_N1_EIGHT_ORIGINAL_HISTORY_NATIVE_COPY_GATE',
            caps=dict(CAPS,main_starts=4,shadow_integrations=29600),CPU_core_s=dict(core=3600,hopper=0),
            wall_s=3600,card=NATIVE_GATE_CARD,states=8,parents=4,trace_only=False,native_gate_only=True)
    raise PermissionError('EXACT_ANCHOR_DYNAMIC_OR_SINGLE_FAILED_TRACE_STAGE')


def validate_window(root,manifest,now):
    root=Path(root);parent=root.parent.parent;stage=read(root/'MP1_STAGE_BINDING.json');spec=specification(stage)
    if root.name!=spec['child'] or parent.name!=n1.PARENT_RUN or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise PermissionError('EXACT_N1_ANCHOR_CPU_SIMULATION_CHILD_NO_CUDA')
    first=parent/n1.FIRST_ADOPTION
    if first.is_symlink() or first.stat().st_mode&0o222 or sha(first)!=n1.ADOPTED_DELTA_SHA:
        raise PermissionError('ORIGINAL_FIRST_N1_ADOPTION_REQUIRED_FOR_ANCHOR_GATE')
    delta=n1.validated_delta(read(first),read(parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'))
    if manifest['execution'].get('anchor_dynamic_n1_binding')!=dict(
            release=n1.RELEASE,adoption_delta_sha256=n1.ADOPTED_DELTA_SHA):
        raise PermissionError('EXPLICIT_N1_DYNAMIC_DIAGNOSTIC_AUTHORITY')
    for name in [spec['card'],INPUTS]:
        p=root/name
        if (p.is_symlink() or p.stat().st_mode&0o222 or not p.resolve().is_relative_to(root.resolve())
                or manifest['execution']['sealed_inputs'].get(name)!=sha(p)):
            raise PermissionError('SEALED_DYNAMIC_CARD_AND_OWNED_INPUTS')
    card=read(root/spec['card'])
    if (card['schema']!=stage['schema'] or card['authority_release']!=n1.RELEASE or card['child']!=spec['child']
            or card['caps']!=spec['caps'] or card['CPU_core_s']!=spec['CPU_core_s'] or card['maximum_wall_s']!=spec['wall_s']
            or card['selected_states']!=spec['states'] or card['distinct_student_parents']!=spec['parents']
            or card['comparison_absolute_tolerance']!=1e-10 or card['maximum_new_GPU_PBS']!=0
            or card['Teacher_queries_or_new_positive_labels'] is not False):
        raise PermissionError('FIXED_EIGHT_ANCHORS_NO_EXPANDED_EXPERIMENT')
    if spec['trace_only'] and (card['only_state']!=FAILED_STATE or card['prior_failure_sha256']!=PRIOR_FAILURE
            or card['single_state_pass_can_qualify_all_eight'] is not False):
        raise PermissionError('ONLY_RECORDED_FAILED_STATE_TRACE_NOT_GATE_PROMOTION')
    if spec.get('ablation_only') and (card['trace02_failure_sha256']!=TRACE02_FAILURE
            or card['branches']!=['FULL_NATIVE_COPY','FRESH_ENGINE_INTEGRATION_RESTORE','NATIVE_COPY_INTEGRATION_RESTORE']
            or card['original_history_shadow_ticks']!=5200 or card['maximum_original_prefix_ticks_per_branch']!=63):
        raise PermissionError('EXACT_SINGLE_STATE_THREE_BRANCH_ENGINEERING_ABLATION')
    if spec.get('native_gate_only'):
        count_key='exact_revalidation_shadow_ticks' if spec.get('endpoint_only') else 'exact_total_shadow_ticks'
        if (card['mechanism_result_sha256']!=COPY_MECHANISM_RESULT
                or card['state_reconstruction']!='COMPLETE_ORIGINAL_ACKNOWLEDGED_HISTORY_THEN_NATIVE_MJDATA_COPY'
                or card['original_history_shadow_ticks']!=27600 or card[count_key]!=29202
                or card['future_consumer_must_reconstruct_and_validate_again'] is not True):
            raise PermissionError('PROVEN_NATIVE_HISTORY_COPY_ROUTE_EIGHT_ANCHORS_ONLY')
    if spec.get('endpoint_only'):
        from .endpoint_ni_delivery import RESULT_SHA,FINAL_SHA
        if (card['actual_NI16_result_sha256']!=RESULT_SHA or card['actual_NI16_job_final_sha256']!=FINAL_SHA
                or card['endpoint_prefixes']!=16 or card['maximum_new_endpoint_shadow_ticks']!=3200
                or card['maximum_total_actual_shadow_ticks']!=32402
                or card['future_consumer_must_reconstruct_and_validate_again'] is not True
                or card['original_safety_reference_gate_unchanged'] is not True):
            raise PermissionError('ONLY_SIXTEEN_ALREADY_GENERATED_MATCHED_ENDPOINT_PREFIXES')
    began=stage['diagnostic_created_epoch'];end=stage['diagnostic_deadline_epoch']
    if (not delta['autonomy_started_epoch']<=began<delta['new_submission_and_candidate_cutoff_epoch']
            or not began<=now<end<=min(began+spec['wall_s'],delta['new_experiment_cutoff_epoch'])):
        raise PermissionError('ORIGINAL_N1_CLOCK_AND_FINITE_DYNAMIC_WINDOW')
    cpus=stage['diagnostic_CPU_ids']
    if not isinstance(cpus,list) or len(cpus)!=1 or type(cpus[0]) is not int or cpus[0]<0:
        raise PermissionError('ONE_PREBOUND_CPU_FOR_DYNAMIC_DIAGNOSTIC')
    return dict(deadline_epoch=end,CPU_ids=cpus,N1_T0_epoch=delta['autonomy_started_epoch'])
