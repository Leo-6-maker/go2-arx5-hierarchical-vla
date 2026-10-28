"""The original N1 maximum16 missing draws, with no replay of policy conditions."""
import json
from pathlib import Path
import re
import signal

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import digest,unseal
from scripts.pi0294.e1 import model
from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown
from scripts.pi0294.native_dataset import sha
from . import n1_limits as n1
from .failure_config import ATTEMPTS,selected_sequences
from .probe_compare import PRECISION

SCHEMA='MP1_N1_CURRENT_EIGHT_ANCHOR_ENDPOINT_NI16_V1'
PLAN='plans/N1/alloc-10/ENDPOINT_ANCHOR_NI16_BINDING.json'
CARD='plans/N1_CURRENT_ANCHOR_ENDPOINT_NI16_V1.json'
GATE='plans/N1/alloc-10/NATIVE_HISTORY_GATE.json'
JOIN='plans/N1/alloc-10/ORIGINAL_EIGHT_RECEIPT_JOIN.private.json'
MATRIX='plans/N1/alloc-10/COMPLETE_ENDPOINT_MATRIX.json'
FILES=[PLAN,CARD,GATE,JOIN,MATRIX]
GATE_RESULT='cfbd6a426e6934140d31940a7463e07003981a1f0c94bcbf09f6c83ff76b94cb'
BASELINE='1cf83ad430c03c1d953266562cf90f3d0475f49c806701448fe5750d494c4d8c'
CHECKPOINTS={'S-BC-2048':'ac0f46924d540ec5519f8c64b1ec0ea1c7f37bba96df0a21744e6a6a8e88efc2',
    'S-CORR-2048':'94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'}
LABEL='N1-CURRENT-EIGHT-ENDPOINT-NI16'


def selected_ids():
    return {identifier+'-S'+str(sequence).zfill(3) for identifier in ATTEMPTS for sequence in selected_sequences(identifier)}


def validate_values(plan,card,gate,join,matrix):
    if (plan['schema']!=SCHEMA or plan['authority_release']!=n1.RELEASE or plan['run_uuid']!='source-pair-01'
            or plan['allocation_number']!=10 or plan['new_queries_max']!=16 or plan['new_training_updates']!=0
            or plan['new_physics']!=0 or plan['prior_generation_queries']!=2390 or plan['prior_formal_updates']!=4096
            or plan['N1_adoption_sha256']!=n1.ADOPTED_DELTA_SHA or plan['N1_T0_epoch']!=1792264081):
        raise PermissionError('EXACT_ORIGINAL_N1_ALLOCATION10_SIXTEEN_DIAGNOSTIC_DRAWS')
    exact=dict(schema=SCHEMA,authority_release=n1.RELEASE,run_uuid='source-pair-01',only_allocation=10,
        states=8,independent_student_parents=4,models=list(CHECKPOINTS),draws_per_model=8,maximum_new_neural_queries=16,
        maximum_new_PBS=1,maximum_allocated_H200_s=7200,original_allocation_closeout_s=2700,
        maximum_new_output_bytes=67108864,maximum_active_neural_CPUs=4,maximum_each_draw_wall_s=60,new_optimizer_updates=0,
        new_live_or_shadow_integrations=0,new_Teacher_labels=0,TRAIN_only=True,
        baseline_checkpoint_manifest_sha256=BASELINE,endpoint_checkpoint_manifest_sha256=CHECKPOINTS,
        native_history_gate_result_sha256=GATE_RESULT,canonical_precision=PRECISION['mode'])
    if any(card.get(k)!=v for k,v in exact.items()):raise PermissionError('EXACT_PUBLISHED_NI16_CARD_NO_EXTRA_MODELS_OR_BUDGET')
    for name in ['actual_new_same_job600s80cycle_qualification_required','actual_generated_noise_fingerprint_must_match_original',
            'same_seed_without_actual_noise_identity_is_insufficient','old_eight_BC512_draws_reused_not_resampled',
            'all_original32_policy_conditions_must_be_terminal_before_load','all24_endpoint_conditions_must_be_independently_audited',
            'per_draw_durable_intent_no_unknown_resampling','future_CPU_shadow_requires_separate_card_and_full_history_revalidation']:
        if card.get(name) is not True:raise PermissionError('ORIGINAL_NI16_SAFETY_AND_EVIDENCE_GATES_REQUIRED')
    if card['automatic_next_allocation'] or card['output_is_closed_loop_or_task_success']:
        raise PermissionError('NO_AUTOMATIC_NEXT_JOB_OR_OFFLINE_CLOSED_LOOP_CLAIM')
    cells=matrix['cells'];attempts=matrix['actual_attempts'];endpoint=[c for c in cells if '-2048-' in c['model']]
    if (matrix['schema']!='MP1_ALL_FIXED_SOURCE_BEHAVIOR_AND_FAILURE_MATRIX_V1' or len(cells)!=32
            or len(endpoint)!=24 or not all(c['valid_policy_outcome'] for c in endpoint) or len(attempts)!=35
            or any(c['status'] in ['PENDING_OR_IN_PROGRESS','NOT_RUN','UNVERIFIED_POLICY_OUTCOME','ACTUAL_SAFETY_FAIL'] for c in cells)
            or matrix['actual_safety_stop_present'] or len({a['attempt_id'] for a in attempts})!=35):
        raise PermissionError('ALL_ORIGINAL32_TERMINAL_AND_ALL24_ENDPOINT_AUDITS_REQUIRED')
    if any(not a['raw_audit_verified'] or not a['cross_host_chain_verified'] or not a['qualified_policy_outcome']
            for a in attempts if '-2048-' in a['model'] and a['status']!='ENGINEERING_INCOMPLETE'):
        raise PermissionError('EACH_ENDPOINT_HAS_ITS_OWN_RAW_AND_FRESH_INPUT_ACTION_ACK_AUDIT')
    if any(matrix['checkpoint_manifest_identities'].get(name)!=value for name,value in CHECKPOINTS.items()):
        raise PermissionError('AUDITED_MATRIX_HAS_THE_EXACT_TWO_ENDPOINT_WEIGHTS')
    if (gate['schema']!='MP1_N1_EIGHT_NATIVE_HISTORY_COPY_ACTUAL_V1' or gate['authority_release']!=n1.RELEASE
            or gate['result_sha256']!=GATE_RESULT or gate['full_eight_anchor_gate_passed'] is not True
            or gate['comparison_absolute_tolerance']!=1e-10 or gate['tolerance_changed'] is not False
            or gate['new_lease']!='CLOSED' or gate['future_consumer_must_reconstruct_and_validate_again'] is not True
            or len(gate['rows'])!=8 or {r['id'] for r in gate['rows']}!=selected_ids()):
        raise PermissionError('ACTUAL_CLOSED_EIGHT_NATIVE_HISTORY_GATE_REQUIRED')
    if (join['status']!='PASS_EIGHT_CURRENT_SBC512_ACTIONS_ORIGINAL_INPUT_NOISE_AND_NATIVE_ARRAY_BINDINGS'
            or join['new_queries'] or join['new_physics'] or join['hopper']['checkpoint_manifest_sha256']!=BASELINE):
        raise PermissionError('ORIGINAL_EIGHT_BASELINE_RECEIPTS_NO_NEW_BASELINE_QUERY')
    rows=join['core']['rows'];native=join['hopper']['rows']
    if (len(rows)!=8 or len(native)!=8 or {r['id'] for r in rows}!=selected_ids()
            or {r['id'] for r in native}!=selected_ids() or len({r['episode_uuid'] for r in rows})!=4
            or len({r['query_uuid'] for r in rows})!=8):
        raise PermissionError('EIGHT_FIXED_STATES_FOUR_PARENTS_ONLY')
    indexed={r['id']:r for r in native};gates={r['id']:r for r in gate['rows']}
    expected_files={}
    for row in rows:
        other=indexed[row['id']];key=row['query_uuid']
        parent=row['id'].rsplit('-S',1)[0];parts=parent.split('-');noise=other['actual_native_noise_fingerprint']
        if (not re.fullmatch('[0-9a-f-]{36}',key) or row['query_uuid']!=other['query_uuid']
                or row['scene_group']!='PI0294/TRAIN/'+'/'.join(parts[-3:-1]) or row['task']!=parts[-1]
                or row['sequence']!=int(row['id'].rsplit('-S',1)[1]) or other['sequence']!=row['sequence']
                or row['observation_sha256']!=gates[row['id']]['observation_sha256']
                or row['observation_fingerprint']!=other['observation_fingerprint']
                or other['checkpoint_manifest_sha256']!=BASELINE
                or any(other['actual_precision'].get(k)!=v for k,v in PRECISION.items())
                or noise.get('shape')!=[1,40,132] or noise.get('dtype')!='torch.float32'
                or re.fullmatch('[0-9a-f]{64}',noise.get('sha256','')) is None):
            raise PermissionError('EXACT_NATIVE_BASELINE_INPUT_NOISE_AND_PRECISION_JOIN')
        base='evidence/S-BC-512-TRAIN4-ALLOCATION-01/resident/'
        expected_files[base+'receipts/'+key+'.json']=other['receipt_sha256']
        expected_files[base+'receipts/'+key+'.npz']=other['arrays_sha256']
    if plan['native_receipt_files']!=expected_files:raise PermissionError('EXACT_SIXTEEN_ORIGINAL_NATIVE_RECEIPT_FILES')
    return dict(rows=rows,native=indexed)


def binding_files(boundary):
    if 'endpoint_probe_binding_relative' not in boundary:return []
    if boundary['target_allocation_number']!=10 or boundary['endpoint_probe_binding_relative']!=PLAN:
        raise PermissionError('NI16_ONLY_EXPLICIT_ALLOCATION10_COLD_BOUNDARY')
    return FILES


def validate_binding(root,boundary,manifest):
    root=Path(root)
    if not binding_files(boundary):return None
    for relative in FILES:
        p=root/relative
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or p.stat().st_mode&0o222
                or manifest['execution']['sealed_inputs'].get(relative)!=sha(p)):
            raise PermissionError('SEALED_NI16_CARD_GATE_JOIN_AND_PLAN')
    plan=read(root/PLAN)
    if any(plan['binding_file_sha256'].get(n)!=sha(root/n) for n in [CARD,GATE,JOIN,MATRIX]):
        raise PermissionError('NI16_BOUND_IMMUTABLE_FILE_HASHES')
    return validate_values(plan,read(root/CARD),read(root/GATE),read(root/JOIN),read(root/MATRIX))


def checked_source_boundary(root,past,outcomes):
    if (past['GPU_allocated_s']!=42088 or past['generation_queries']!=2390 or past['optimizer_updates']!=4096
            or past['pending_updates'] or past['GPU_reserved_s'] or len(past['PBS_submissions'])!=9 or len(outcomes)!=35):
        raise PermissionError('NI16_ONLY_AFTER_ACTUAL_TERMINAL_COMPLETE_SOURCE09_BASELINE')
    matrix=read(Path(root)/MATRIX);indexed={a['attempt_id']:a for a in matrix['actual_attempts']}
    if len(indexed)!=35 or {r['attempt_id'] for r in outcomes}!=set(indexed):
        raise PermissionError('NI16_ALL_ORIGINAL35_OUTCOMES_PRESERVED')
    for row in outcomes:
        actual=indexed[row['attempt_id']]
        fields=['attempt_id','status','failure_stage','model','task','split','scene_group','duration_s',
            'main_integrations','RGB_pairs','S1_rejections','live_native_failures','teacher_takeover']
        if (any(row[k]!=actual[k] for k in fields)
                or row['pipeline_trustworthy']!=actual['adjudication_pipeline_trustworthy']
                or row.get('result_sha256')!=actual.get('raw_result_sha256')):
            raise PermissionError('NI16_RAW_ADJUDICATIONS_MUST_MATCH_THE_INDEPENDENT_AUDITED_MATRIX')


def current(root):
    from .cold_boundary import validate
    root=Path(root);m=read(root/'RUN_MANIFEST.json');boundary=validate(root)
    if boundary is None or not binding_files(boundary):return None
    value=validate_binding(root,boundary,m);value['plan']=read(root/PLAN);return value


def original_payload(root,row,native):
    root=Path(root);d=root/'evidence/S-BC-512-TRAIN4-ALLOCATION-01/resident';key=row['query_uuid']
    request=read(d/'inbox'/(key+'.json'));receipt=d/'receipts'/(key+'.json')
    if (sha(receipt)!=native['receipt_sha256'] or sha(d/'receipts'/(key+'.npz'))!=native['arrays_sha256']
            or request['sha256']!=row['request_sha256']):
        raise PermissionError('ORIGINAL_NI16_INPUT_OR_NOISE_RECEIPT_CHANGED')
    receipt_value=read(receipt);identity=read(d/'STATE.json')['identity']
    if (receipt_value['seed']!=native['noise_seed'] or receipt_value['noise']!=native['actual_native_noise_fingerprint']
            or receipt_value['inference_precision']!=native['actual_precision']
            or receipt_value['physics_steps']!=0 or receipt_value['extra_correction'] is not False
            or identity['global_step']!=512 or identity['checkpoint_manifest_sha256']!=BASELINE):
        raise PermissionError('ACTUAL_BASELINE_SEED_NOISE_PRECISION_AND_CHECKPOINT_RECEIPT')
    body=unseal(request);payload=body['payload']
    if (body['query_uuid']!=key or body['observation_sha256']!=row['observation_sha256']
            or body['sequence']!=row['sequence'] or body['tick']!=row['sequence']*200
            or body['identity']!=identity['relay_identity']
            or digest({k:v for k,v in payload.items() if k!='snapshot_uuid'})!=native['observation_fingerprint']):
        raise PermissionError('EXACT_ORIGINAL_PROCESSOR_PAYLOAD_IDENTITY')
    return payload


def run(root,request,ledger,guard,load,unload,instance):
    root=Path(root);binding=current(root)
    if binding is None or request['submit_count']!=10 or request['walltime_s']>7200:
        raise PermissionError('EXPLICIT_BOUND_SINGLE_NI16_NATIVE_ALLOCATION')
    from .pre_model_qualification import require
    require(root,root/'pbs/alloc-10',read(root/'pbs/alloc-10/ALLOCATION_ADMISSION.json'))
    before=read(ledger.path)
    if before['generation_queries']!=2390 or before['optimizer_updates']!=4096 or before['pending_updates']:
        raise PermissionError('ORIGINAL_NI16_START_COUNTERS_NO_REPEAT')
    output=root/'evidence'/LABEL;output.mkdir(mode=0o700);records=[]
    for name,manifest in CHECKPOINTS.items():
        branch=name.removesuffix('-2048');checkpoint=root/'training'/branch/'checkpoint-2048'
        if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=manifest:raise PermissionError('ORIGINAL2048_ENDPOINT_CHECKPOINT')
        # No dataset initialization, optimization or live resident service.
        guard();proof=load(checkpoint,LABEL+'-'+name,False);write(output/(name+'-FROZEN.json'),proof)
        folder=output/name;folder.mkdir(mode=0o700)
        try:
            for row in binding['rows']:
                guard();native=binding['native'][row['id']];payload=original_payload(root,row,native)
                target=folder/(row['id']+'.npz')
                def expire(signum,frame):raise TimeoutError('ORIGINAL60SECOND_NI_DRAW_LIMIT_NO_RESAMPLING')
                previous=signal.signal(signal.SIGALRM,expire);signal.alarm(60)
                try:physical,meta=model.fixed_draw(root,LABEL+'-'+name,row['id'],instance(),payload,native['noise_seed'],target,ledger)
                finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
                if (meta['noise']!=native['actual_native_noise_fingerprint']
                        or any(meta['inference_precision'].get(k)!=v for k,v in PRECISION.items())):
                    raise PermissionError('ACTUAL_GENERATED_NOISE_AND_CANONICAL_PRECISION_MUST_MATCH_BASELINE')
                original_payload(root,row,native)
                if physical.shape!=(40,10) or not np.isfinite(physical).all():raise ValueError('ORIGINAL_FINITE_EEF10_H40_OUTPUT')
                records.append(dict(id=row['id'],model=name,arrays_sha256=meta['arrays_sha256'],
                    original_query_uuid=row['query_uuid'],observation_fingerprint=native['observation_fingerprint'],
                    original_observation_sha256=row['observation_sha256'],noise_seed=native['noise_seed'],
                    noise=meta['noise'],inference_precision=meta['inference_precision'],
                    checkpoint_manifest_sha256=manifest,new_query=True,live_or_shadow_integrations=0))
                write(output/'PROGRESS.json',dict(completed=len(records),records=records))
        finally:unload()
    after=read(ledger.path)
    if len(records)!=16 or after['generation_queries']!=2406 or after['optimizer_updates']!=4096 or after['pending_updates']:
        raise PermissionError('EXACT_SIXTEEN_NEW_DRAWS_NO_OPTIMIZER_OR_PHYSICS')
    result=dict(status='COMPLETE_SIXTEEN_SAME_INPUT_NOISE_ENDPOINT_DRAWS_NOT_POLICY_EVALUATION',
        records=records,new_neural_queries=16,original_eight_BC512_draws_reused=True,
        native_history_gate_result_sha256=GATE_RESULT,new_optimizer_updates=0,new_physics=0,
        closed_loop_or_task_success_claimed=False,CPU_shadow='NOT_RUN_REQUIRES_SEPARATE_HISTORY_REPLAY_BINDING')
    write(output/'RESULT.json',result);return result


def closed_draws(root):
    root=Path(root);path=root/'evidence'/LABEL/'RESULT.json'
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('ONLY_OWNED_NI16_RESULT_NO_SYMLINK')
    if not path.exists():
        if (root/'evidence'/LABEL).exists():raise MutationOutcomeUnknown('NI16_UNFINISHED_QUERY_SCOPE_NO_RESAMPLING')
        return 0
    from .n1_boundary import paths
    binding=validate_binding(root,read(root/paths(10)[0]),read(root/'RUN_MANIFEST.json'))
    if binding is None:raise PermissionError('NI16_RESULTS_REQUIRE_THEIR_IMMUTABLE_COLD_BINDING')
    value=read(path);records=value['records'];expected={(name,identifier) for name in CHECKPOINTS for identifier in selected_ids()}
    if (value['status']!='COMPLETE_SIXTEEN_SAME_INPUT_NOISE_ENDPOINT_DRAWS_NOT_POLICY_EVALUATION'
            or len(records)!=16 or {(r['model'],r['id']) for r in records}!=expected or value['new_neural_queries']!=16
            or value['new_optimizer_updates'] or value['new_physics'] or value['closed_loop_or_task_success_claimed']):
        raise PermissionError('EXACT_COMPLETED_NI16_ARRAY_AND_COUNT_CLOSURE')
    journal_root=root/'reservations/probe-generations'
    if not journal_root.is_dir() or journal_root.is_symlink():
        raise MutationOutcomeUnknown('NI16_MISSING_OR_REDIRECTED_ORIGINAL_QUERY_JOURNAL')
    journal=Journal(journal_root,root.name)
    expected_messages={'probe-'+LABEL+'-'+name+'-'+identifier for name,identifier in expected}
    actual_messages={p.stem for p in journal.directory.glob('probe-'+LABEL+'-*.json')}
    expected_arrays={path.parent/name/(identifier+'.npz') for name,identifier in expected}
    if actual_messages!=expected_messages or set(path.parent.glob('*/*.npz'))!=expected_arrays:
        raise MutationOutcomeUnknown('NI16_UNDECLARED_OR_MISSING_QUERY_INTENTS_OR_ARRAYS')
    for row in records:
        target=path.parent/row['model']/(row['id']+'.npz');original=binding['native'][row['id']]
        if (target.is_symlink() or not target.resolve().is_relative_to(path.parent.resolve())
                or sha(target)!=row['arrays_sha256'] or row['noise']!=original['actual_native_noise_fingerprint']
                or row['noise_seed']!=original['noise_seed'] or row['original_query_uuid']!=original['query_uuid']
                or row['observation_fingerprint']!=original['observation_fingerprint']
                or row['checkpoint_manifest_sha256']!=CHECKPOINTS[row['model']]
                or row['new_query'] is not True or row['live_or_shadow_integrations']!=0
                or any(row['inference_precision'].get(k)!=v for k,v in PRECISION.items())):
            raise PermissionError('NI16_RAW_ARRAY_AND_ORIGINAL_NOISE_HASH')
        label=LABEL+'-'+row['model'];checkpoint=root/'training'/row['model'].removesuffix('-2048')/'checkpoint-2048'
        if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=CHECKPOINTS[row['model']]:
            raise PermissionError('NI16_ORIGINAL_ENDPOINT_WEIGHT_MANIFEST_CHANGED')
        complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
        files={n:s for n,s in complete['files'].items() if n.endswith('.safetensors')
            or n in ['config.json','model.safetensors.index.json','statistics.json'] or n.startswith('processor/')}
        weights=digest(files)
        args=dict(label=label,id=row['id'],seed=original['noise_seed'],observation_fingerprint=original['observation_fingerprint'],
            checkpoint_files_fingerprint=weights,precision=PRECISION['mode'])
        meta=journal.completed('probe-'+label+'-'+row['id'],args)
        if (meta is None or meta['arrays_sha256']!=row['arrays_sha256'] or meta['noise']!=row['noise']
                or meta['inference_precision']!=row['inference_precision']):
            raise MutationOutcomeUnknown('NI16_EXACT_QUERY_INTENT_NOT_COMPLETE')
        with np.load(target,allow_pickle=False) as arrays:
            if (arrays['physical_H40'].shape!=(40,10) or arrays['normalized'].shape!=(1,40,132)
                    or not np.isfinite(arrays['physical_H40']).all() or not np.isfinite(arrays['normalized']).all()
                    or json.loads(arrays['noise_fingerprint_json'].item())!=row['noise']
                    or arrays['observation_fingerprint'].item()!=row['observation_fingerprint']
                    or arrays['checkpoint_files_fingerprint'].item()!=weights
                    or int(arrays['noise_seed'].item())!=original['noise_seed']
                    or any(json.loads(arrays['inference_precision_json'].item()).get(k)!=v for k,v in PRECISION.items())):
                raise PermissionError('ACTUAL_NI16_NORMALIZED_PHYSICAL_NOISE_AND_INPUT_ARRAYS')
    return 16
