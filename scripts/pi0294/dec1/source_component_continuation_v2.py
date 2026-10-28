"""Pure aggregation of pinned independent audits; never dispatch or settlement.

Use existing source_component_terminal.audit for EACH child first. This module
only joins those read-only audit receipts through one owner-predeclared cold
manifest binding. It never repairs the historical UNKNOWN or native artifacts.
"""
from collections import Counter
from copy import deepcopy
from scripts.pi0294.dec1.source_component_terminal import document,sha,require
from scripts.pi0294.dec1 import source_commissioning_plan as plans
from scripts.pi0294.dec1.source_cpu_intake import PLAN_SHA256,PLAN_OWNER_RELEASE,ROLE

OLD_REQUEST='23da1121-b1cd-4fef-83b4-a28e45e108b1'
OLD_RUN='dec1-source-component-01'
OLD_SOURCE='55e73c7b6e2c6584f90b14d64ef9f5b842fd2e2c'
OLD_MANIFEST='cfc054345e48b5dfe9a4b9f82a1f0b8db22c72c25183affe04cde4a0760232bd'
OLD_AUDIT='d12895ad02bafbeccbc48ae03064292fbca20245df6bdd6abf93424b87598803'
OLD_CLOSURE='46bfdc1ed854483b618904f5ce787178d0aabcb6b8d06df200c5b4d581edd214'
OLD_SETTLEMENT='f27cb94173dbb751bb14f4c5172a1141e72c43f54aeaec64926abd9111014283'
COLD_SOURCE='47dc6001227572f031ebc3cc33d56a487027d8cb'
COLD_REQUEST='081c5299-5594-4ff6-8676-d48b9652cf52'
COLD_MANIFEST='14ce2f841c8db3de7e2eb9c191e95fc9b26720f628493774e4fd954012556b0f'
KNOWN_SCHEMA='DEC1_SOURCE_CPU_KNOWN_TERMINAL_SETTLEMENT_V1'
KNOWN_INDEX='source_cpu_known_terminal_settlements'
HISTORY_PLAN='plans/SOURCE_CLOSED_FINANCIAL_HISTORY.private.json'
REPAIR_PLAN='plans/SOURCE_ENGINEERING_FIX_EVIDENCE.json'
REPAIR_SHA='28221c1935f960b51945516924088b76d414cbaa26eafb9c2e208fe3f766d8ba'
UNKNOWN='ENGINEERING_INCOMPLETE_UNKNOWN'
STATUSES={'NOT_RUN',UNKNOWN,'KNOWN_REFERENCE_OR_S1_REJECT','NATIVE_SAFETY_FAIL','COMPONENT_COMPLETE'}
PURPOSE='SOURCE_INTERFACE_COMPONENT_QUALIFICATION_ONLY'
BINDING_SCHEMA='DEC1_SOURCE_COMPONENT_FINITE_CONTINUATION_BINDING_V1'


def _receipt(raw,pin):
    require(sha(raw)==pin,'PINNED_READONLY_RECEIPT_BYTES_CHANGED')
    envelope=document(raw)
    require(type(envelope['returncode']) is int and envelope['returncode']==0
        and envelope['read_only'] is True,'ACTUAL_SUCCESSFUL_READONLY_RECEIPT_REQUIRED')
    return document(envelope['stdout'])


def _audit(raw,pin,source,ids):
    receipt=_receipt(raw,pin);value=receipt['result']
    require(receipt['source_sha']==source and receipt['readonly'] is True
        and type(receipt['new_native_integrations']) is int and receipt['new_native_integrations']==0
        and receipt['denied_external_native_attempts']==[],'ORIGINAL_INDEPENDENT_BYTE_AUDIT_REQUIRED')
    rows=value['conditions']
    require(value['schema']=='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1'
        and value['purpose']==PURPOSE and value['student_neural_ACK_qualification'] is False
        and value['target_qualification'] is False and value['core64_cells']==0
        and value['new_training_data'] is False and [r['condition_id'] for r in rows]==ids
        and all(r['status'] in STATUSES for r in rows)
        and value['component_counts']==dict(Counter(r['status'] for r in rows)),
        'EXACT_FOUR_COMPONENT_ONLY_TERMINAL_ROWS_REQUIRED')
    return rows


def aggregate(plan_bytes,population_bytes,original_audit_bytes,closure_bytes,settlement_bytes,*,
              cold_audit_bytes=None,cold_audit_sha256=None,cold_manifest_bytes=None,
              cold_manifest_sha256=None,cold_budget_bytes=None,cold_audit_result_bytes=None,
              known_settlement_bytes=None,known_settlement_sha256=None,once_pairs=None):
    """Cold inputs are all absent (baseline) or actual dual-settled receipts, exactly once.

    The original budget/continuation is sealed in the already-fixed Source02 manifest;
    this pure function does not create, admit or spend it. Reuse the original
    terminal auditor and native read-only adapter; no second scientific runner.
    """
    require(sha(plan_bytes)==PLAN_SHA256,'UNCHANGED_FIXED_FOUR_CONDITION_PLAN_REQUIRED')
    plan=plans.validate(document(plan_bytes),population_bytes,owner_release=PLAN_OWNER_RELEASE)
    ids=[c['condition_id'] for c in plan['conditions']]
    original=_audit(original_audit_bytes,OLD_AUDIT,OLD_SOURCE,ids)
    require([r['status'] for r in original]==[UNKNOWN,'NOT_RUN','NOT_RUN','NOT_RUN'],
        'ORIGINAL_FAILED_ATTEMPT_AND_THREE_UNSTARTED_CONDITIONS_PRESERVED')
    closed=_receipt(closure_bytes,OLD_CLOSURE)
    require(sha(settlement_bytes)==OLD_SETTLEMENT,'ORIGINAL_FINANCIAL_SETTLEMENT_BYTES_CHANGED')
    invoice=document(settlement_bytes)['invoice']
    require(closed['status']=='INDEPENDENT_DUAL_FINANCIAL_CLOSURE_RETAINED_UNKNOWN_AND_FILES_CONFIRMED'
        and closed['source_lease_status']=='CLOSED' and closed['active_workers']==0
        and closed['original_child_ledger_unchanged'] is True
        and closed['unknown_and_original_live_reservation_preserved'] is True
        and closed['scientific_verdict']==UNKNOWN and closed['actual_STOP_ACK'] is False
        and closed['new_writes']==closed['new_native_integrations']==0
        and invoice['request_id']==OLD_REQUEST and invoice['child_run_uuid']==OLD_RUN
        and invoice['runtime_source']==OLD_SOURCE and invoice['child_manifest_sha256']==OLD_MANIFEST
        and invoice['scientific_verdict']==UNKNOWN and invoice['live_financial_upper']==16000
        and invoice['financial_charged']['physics_ticks']==16490,
        'CLOSED_FINANCES_DO_NOT_RESOLVE_ORIGINAL_SCIENTIFIC_UNKNOWN')
    optional=(cold_audit_bytes,cold_audit_sha256,cold_manifest_bytes,cold_manifest_sha256,
              cold_budget_bytes,cold_audit_result_bytes,known_settlement_bytes,known_settlement_sha256,once_pairs)
    require(all(v is None for v in optional) or all(v is not None for v in optional),
        'PARTIAL_COLD_ACTUAL_RECEIPTS_REJECTED')
    cold=None;new_run=None;known=None
    if cold_audit_bytes is not None:
        require(sha(cold_manifest_bytes)==cold_manifest_sha256==COLD_MANIFEST
            and sha(known_settlement_bytes)==known_settlement_sha256,'EXACT_SOURCE02_AND_SETTLEMENT_BYTES')
        manifest=document(cold_manifest_bytes);budget=document(cold_budget_bytes)
        known=document(known_settlement_bytes);new_run=manifest['run_uuid']
        require(manifest['scope']==ROLE and manifest['code']['vla_commit']==COLD_SOURCE
            and isinstance(new_run,str) and new_run and new_run!=OLD_RUN
            and manifest['data']['manifest_sha256']==PLAN_SHA256
            and manifest['execution']['sealed_inputs']['plans/DEC1_SOURCE_CPU_BUDGET.json']==sha(cold_budget_bytes),
            'ACTUAL_FROZEN_SOURCE02_BUDGET_AND_PLAN')
        binding=budget['continuation'];sealed=manifest['execution']['sealed_inputs']
        require(budget['request_id']==COLD_REQUEST and budget['requested']['episode_starts']==4
            and type(binding['original_commissioning_requested']) is int and binding['original_commissioning_requested']==3
            and type(binding['engineering_replacements_requested']) is int and binding['engineering_replacements_requested']==1
            and binding['replacement_of_request_id']==OLD_REQUEST and binding['replacement_condition_id']==ids[0]
            and binding['replacement_index']==1 and type(binding['replacement_index']) is int
            and binding['original_condition_ids']==ids[1:]
            and binding['history_sha256']==sealed[HISTORY_PLAN]
            and binding['engineering_fix_evidence_sha256']==sealed[REPAIR_PLAN]==REPAIR_SHA,
            'ORIGINAL3_PLUS_UNIQUE_ENGINEERING1_NO_VALID_FAILURE_REPLACEMENT')
        cold=_audit(cold_audit_bytes,cold_audit_sha256,COLD_SOURCE,ids)
        raw_result=document(cold_audit_result_bytes)
        wrapped_result=_receipt(cold_audit_bytes,cold_audit_sha256)['result']
        require(raw_result==wrapped_result and known['schema']==KNOWN_SCHEMA
            and known['request_id']==COLD_REQUEST and known['settlement_id']==COLD_REQUEST+'-known-terminal-close'
            and known['manifest_sha256']==COLD_MANIFEST and known['child_run_uuid']==new_run
            and known['budget_sha256']==sha(cold_budget_bytes) and known['plan_sha256']==PLAN_SHA256
            and known['source_sha']==COLD_SOURCE and known['audit_sha256']==sha(cold_audit_result_bytes)
            and known['conditions']==cold and known['student_neural_ACK_qualification'] is False
            and known['target_qualification'] is False and known['core64_cells']==0
            and known['new_training_data'] is False and known['native_dispatch_admitted'] is False,
            'KNOWN_SETTLEMENT_MUST_BIND_EXACT_ACTUAL_AUDIT_SOURCE_CONDITIONS')
        attempted=[r for r in cold if r['status']!='NOT_RUN']
        require(all(r['status']!=UNKNOWN for r in attempted)
            and [r['condition_id'] for r in known['attempts']]==[r['condition_id'] for r in attempted]
            and all(x['status']==r['status'] and x['actual_stop_returned'] is True
                and x['unknown_effects']==[] for x,r in zip(known['attempts'],attempted)),
            'ALL_ACTUAL_KNOWN_ATTEMPTS_RETAINED_NO_NOTRUN_CHARGE')
        live=sum(r['main_integrations'] for r in attempted);shadow=sum(r['shadow_integrations'] for r in attempted)
        require(known['actual_live_ticks']==live and known['actual_shadow_ticks']==shadow
            and known['financial_charged']['physics_ticks']==live+shadow
            and known['financial_charged']['episode_starts']==len(attempted),
            'KNOWN_FINANCE_AND_ACTUAL_DOMAINS_MATCH')
        require(type(once_pairs) is dict and set(once_pairs)=={'parent','Core'},'ACTUAL_DUAL_ONCE_RECEIPTS_REQUIRED')
        targets=[]
        for pair in once_pairs.values():
            require(type(pair) is dict and set(pair)=={'ledger_bytes','intent_bytes','receipt_bytes'},
                'EXACT_ONCE_PAIR_BYTES')
            ledger=document(pair['ledger_bytes']);intent=document(pair['intent_bytes']);ack=document(pair['receipt_bytes'])
            require(intent['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
                and ack['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
                and ack['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
                and ack['request_id']==intent['request_id']==known['settlement_id']
                and ack['target']==intent['target'] and ack['target'].startswith('/')
                and ack['before_sha256']==intent['before_sha256']
                and ack['after_sha256']==intent['proposed_sha256']==sha(pair['ledger_bytes'])
                and ack['intent_sha256']==sha(pair['intent_bytes'])
                and intent['native_dispatch_admitted'] is ack['native_dispatch_admitted'] is False
                and ledger[KNOWN_INDEX][COLD_REQUEST]==known
                and ledger['source_cpu_reservations'][COLD_REQUEST]['status']=='SETTLED'
                and ledger['source_cpu_reservations'][COLD_REQUEST]['known_terminal_settlement_sha256']==known_settlement_sha256
                and ledger['source_cpu_reservations'][COLD_REQUEST]['continuation']==binding
                and ledger['source_cpu_financial_closeouts'][OLD_REQUEST]==invoice,
                'ACTUAL_DUAL_WRITE_READBACK_NOT_A_PURE_PROPOSAL')
            targets.append(ack['target'])
        require(targets[0]!=targets[1],'PARENT_AND_CORE_ARE_DISTINCT_ACTUAL_RECEIPTS')
    attempts=[];views=[]
    for i,row in enumerate(original):
        prior=deepcopy(row);following=None if cold is None else deepcopy(cold[i])
        if prior['status']!='NOT_RUN':
            attempts.append(dict(role='ORIGINAL',run_uuid=OLD_RUN,audit_sha256=OLD_AUDIT,result=deepcopy(prior)))
        role='ENGINEERING_REPLACEMENT' if i==0 else 'FIRST_ATTEMPT_OF_PREVIOUSLY_NOT_RUN'
        if following is not None and following['status']!='NOT_RUN':
            attempts.append(dict(role=role,run_uuid=new_run,audit_sha256=cold_audit_sha256,result=deepcopy(following)))
        current=prior if following is None or following['status']=='NOT_RUN' else following
        views.append(dict(condition_id=ids[i],interface=plan['conditions'][i]['interface'],
            original=prior,cold=following,cold_role=role,current_terminal_status=current['status']))
    counts=Counter(a['result']['status'] for a in attempts)
    return dict(schema='DEC1_SOURCE_COMPONENT_FINITE_CONTINUATION_SUMMARY_V2',purpose=PURPOSE,
        cold_known_settlement_sha256=known_settlement_sha256,
        component_plan_sha256=PLAN_SHA256,original_audit_sha256=OLD_AUDIT,closed_receipt_sha256=OLD_CLOSURE,
        original_scientific_unknown_preserved=True,original_child_ledger_and_files_unchanged=True,
        original_live_financial_upper_preserved=16000,original_financial_charge_ticks_preserved=16490,attempt_records=attempts,conditions=views,
        attempted_component_counts=dict(counts),attempts_observed=len(attempts),
        current_condition_counts=dict(Counter(v['current_terminal_status'] for v in views)),
        observed_engineering_replacements=sum(a['role']=='ENGINEERING_REPLACEMENT' for a in attempts),
        known_reference_or_S1_rejections=counts['KNOWN_REFERENCE_OR_S1_REJECT'],
        native_safety_failures=counts['NATIVE_SAFETY_FAIL'],engineering_incomplete_unknown=counts[UNKNOWN],
        student_neural_ACK_qualification=False,target_WBC_qualification=False,core64_cells=0,
        new_training_data=False,native_dispatch_admitted=False)
