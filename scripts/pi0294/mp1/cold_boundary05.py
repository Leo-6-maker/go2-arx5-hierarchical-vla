"""Exactly one human-authorized fifth slot, preserving the fourth boundary."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha

BINDING='plans/ALLOC05_COLD_BOUNDARY.json'
HISTORY='plans/ALLOC05_PRIOR_RUN_MANIFEST.json'
LEDGER='plans/ALLOC05_PRIOR_HOPPER_LEDGER.json'
EFFECTS='plans/ALLOC05_PARTIAL_EFFECT_CLOSURE.json'
OUTCOMES='plans/ALLOC05_PRIOR_ADJUDICATIONS.json'
AUTHORIZATION='plans/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION.json'
CARD_SHA='5071c5a72b8000eff603e963e82b982eb377d9107f7249de115292db2232d549'
OLD_SOURCE='8e50bcc23318535cd52c386ae862109ae0504f7f'
STOP04_SHA='67da1d7ade8d0dc0a2c07c667f38477cf858dc2621c3f5f0882348a57feb7244'
NEW_STOP='RELAY_ENGINEERING_STOP_ALLOCATION05.json'
REQUIRED=[BINDING,HISTORY,LEDGER,EFFECTS,OUTCOMES,AUTHORIZATION,
    'RELAY_ENGINEERING_STOP.json','RELAY_ENGINEERING_STOP_ALLOCATION04.json']


def validate(root,worker_manifest=None,*,proposed_manifest=None):
    from . import cold_boundary as prior
    root=Path(root);m=read(root/'RUN_MANIFEST.json') if proposed_manifest is None else proposed_manifest
    if root.name!='source-pair-01' or m['execution'].get('cold_engineering_binding_relative')!=BINDING:
        raise PermissionError('EXACT_FIFTH_COLD_SCOPE_ONLY')
    for n in REQUIRED:
        p=root/n
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or p.stat().st_mode&0o222
                or m['execution']['sealed_inputs'].get(n)!=sha(p)):
            raise PermissionError('SEALED_FIFTH_BOUNDARY_EVIDENCE')
    if sha(root/AUTHORIZATION)!=CARD_SHA:raise PermissionError('EXACT_DIRECT_HUMAN_FIFTH_CARD')
    q=read(root/BINDING);old=read(root/HISTORY);past=read(root/LEDGER);proof=read(root/EFFECTS);rows=read(root/OUTCOMES)
    a=read(root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    prior.validate(root,proposed_manifest=old)
    if (q['schema']!='MP1_SAME_ROOT_COLD_ALLOC05_BOUNDARY_V1' or q['authority_release']!=prior.AUTH
            or q['run_uuid']!=root.name or q['parent_run_uuid']!=root.parent.parent.name
            or q['original_T0_epoch']!=1792178085 or a['T0_epoch']!=q['original_T0_epoch']
            or q['target_allocation_number']!=5 or q['effective_source_PBS_cap']!=5
            or q['new_source_sha']!=m['code']['vla_commit'] or old['code']['vla_commit']!=OLD_SOURCE
            or q['prior_manifest_sha256']!=sha(root/HISTORY) or q['authorization_card_sha256']!=CARD_SHA
            or q['new_budget_or_clock'] is not False or q['new_attempt_rights'] is not False
            or q['old_STOP_removed'] is not False or q['automatic_sixth_submission'] is not False
            or sha(root/'RELAY_ENGINEERING_STOP.json')!=prior.STOP_SHA
            or sha(root/'RELAY_ENGINEERING_STOP_ALLOCATION04.json')!=STOP04_SHA):
        raise PermissionError('ONLY_EXACT_HUMAN_FIFTH_SLOT_ORIGINAL_CLOCK_AND_STOPS')
    if ({k:v for k,v in m.items() if k not in ['code','execution']}
            !={k:v for k,v in old.items() if k not in ['code','execution']}):
        raise PermissionError('FIFTH_BOUNDARY_CANNOT_CHANGE_SCIENCE')
    if ({k:v for k,v in m['execution'].items() if k not in ['sealed_inputs','cold_engineering_binding_relative']}
            !={k:v for k,v in old['execution'].items() if k not in ['sealed_inputs','cold_engineering_binding_relative']}):
        raise PermissionError('FIFTH_BOUNDARY_CANNOT_CHANGE_ENTRY')
    variable={'vla_commit','capsule_relative','source_seal_relative','source_seal_sha256','unchanged_learning_files'}
    if ({k:v for k,v in m['code'].items() if k not in variable}!={k:v for k,v in old['code'].items() if k not in variable}):
        raise PermissionError('FIFTH_BOUNDARY_CANNOT_CHANGE_FROZEN_OR_UPSTREAM_CODE')
    expected=dict(old['execution']['sealed_inputs'],**{n:sha(root/n) for n in REQUIRED})
    if m['execution']['sealed_inputs']!=expected:raise PermissionError('FIFTH_BOUNDARY_PRESERVES_ALL_INPUTS')
    if (past['run_uuid']!=root.name or past['GPU_allocated_s']!=24119 or past['GPU_reserved_s']
            or past['pending_updates'] or past['optimizer_updates']!=2560 or past['generation_queries']!=1299
            or [(v['job_id'],v['status'],v['GPU_conservative_charge_s']) for v in past['PBS_submissions']]!=[
                ('672080.hopper-m-02','NATIVE_TERMINAL',11795),('672471.hopper-m-02','NATIVE_TERMINAL',11521),
                ('672829.hopper-m-02','NATIVE_TERMINAL',412),('672907.hopper-m-02','NATIVE_TERMINAL',391)]):
        raise PermissionError('ALL_FOUR_PRIOR_NATIVE_COSTS_APPLY')
    if (proof['status']!='PASS_SOURCE04_KNOWN_PARTIAL_EFFECTS_CLOSED_NOT_POLICY_RESULT'
            or proof['valid_policy_outcome'] is not False or proof['rerun_authority'] is not False
            or proof['old_and_allocation04_STOP_markers_modified'] is not False
            or proof['new_model_queries'] or proof['new_physics']
            or proof['all_local_mutation_journal_entries_complete']!=1970
            or proof['core']['executed_ACKs']!=21 or proof['core']['integrations']!=4200
            or proof['core']['distinct_snapshots']!=22 or proof['core']['raw_camera_hashes']!=44
            or proof['hopper']['native_draws']!=22 or proof['hopper']['executed_draws']!=21
            or proof['hopper']['uninstalled_draws']!=1 or proof['hopper']['all_array_hashes_and_raw_actions_pass'] is not True
            or proof['hopper']['both_STOP_bytes_preserved'] is not True
            or proof['hopper']['uninstalled']!=dict(sequence=21,query_uuid='a9adf6fc-582c-40d0-8321-a6a8c3bf5cce',
                request_sha256='f0ec9f5d89a0577975a4a25c717a56d32185eef1daf72d03316d136d88cea04d',
                action_sha256='a46ff4af267cbdb9ed8a6c1c0e024a58a63f7f5b90b1e20f46ab931eb9a943fa',
                arrays_sha256='0de1e2cc97dcc2ab7c1f296635e83dc9e35f91ebd7ce6dd9344734107e082f14',installed_or_executed=False)):
        raise PermissionError('ACTUAL_21_EXECUTED_AND_ONE_UNINSTALLED_ACTION_CLOSED')
    if len(rows)!=21 or len({v['attempt_id'] for v in rows})!=21 or any(v['live_native_failures'] for v in rows):
        raise PermissionError('ALL_21_PRIOR_OUTCOMES_AND_NO_NATIVE_FAULT')
    from .plans import remaining_panel
    plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json')
    remaining=[c for spec in plan['panels'].values() for c in remaining_panel(rows,spec)]
    corr=[c for c in remaining if c['mp1_model_label']=='S-CORR-512-TRAIN4']
    if (len(remaining)!=13 or len(corr)!=1 or corr[0]['id']!='PI0294-MP1-S-CORR-512-TRAIN4-F1-000-T2-REPLACEMENT-01'):
        raise PermissionError('ONLY_ORIGINAL_F1_T2_REPLACEMENT_NO_F1_T1_THIRD')
    hashes=old['code']['unchanged_learning_files']
    for manifest in [m]+([worker_manifest] if worker_manifest is not None else []):
        sealpath=root/manifest['code']['source_seal_relative'];seal=read(sealpath)
        if (manifest['code'].get('unchanged_learning_files')!=hashes or sha(sealpath)!=manifest['code']['source_seal_sha256']
                or any(seal['files'].get(n)!=digest for n,digest in hashes.items())):
            raise PermissionError('FIFTH_BOUNDARY_ORIGINAL_LEARNING_AND_PLANT_BYTES')
    return q


def effective_PBS_cap(root):
    root=Path(root);m=read(root/'RUN_MANIFEST.json')
    if m.get('execution',{}).get('cold_engineering_binding_relative')!=BINDING:return 4
    validate(root)
    return 5
