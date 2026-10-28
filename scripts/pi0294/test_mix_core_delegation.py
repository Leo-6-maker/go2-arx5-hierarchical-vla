"""Finite delegation CPU contracts, not native model/Eval qualification.

The original complete-budget fixture uses captured historical accounting and
population inputs. Future request, Core process/device and terminal fields
remain explicit fixtures. Only the temporary onceIO files are actual writes.
"""
from copy import deepcopy
import json

import pytest

from scripts.pi0294.test_v5_core_full_admission import full, captured_global
from scripts.pi0294.pv1 import mix_accounting as a, mix_core_accounting as c
from scripts.pi0294.pv1 import mix_core_delegation as d, mix_core_settlement as terminal
from scripts.pi0294.pv1 import successor_cd as s
from scripts.pi0294.pv1.successor_core_context import LEDGER_SCHEMA
from scripts.pi0294.dec1 import source_cpu_once_io as once


def selected(full):
    """New cold selector; never mutate an old running or failed capsule."""
    v = deepcopy(full)
    m, r, g, l, p, snap = (json.loads(v[k]) for k in ('manifest_bytes', 'request_bytes',
        'global_bytes', 'local_bytes', 'native_facts_bytes', 'snapshot_bytes'))
    m['execution']['global_accounting_mode'] = d.MODE
    r['global_accounting_mode'] = d.MODE; r['run_manifest_sha256'] = a.sha(a.raw(m))
    p.update(request_sha256=a.sha(a.raw(r)), run_manifest_sha256=a.sha(a.raw(m)))
    for obj in (g, l):
        obj[c.FIELD][r['request_id']].update(global_accounting_mode=d.MODE,
            request_sha256=a.sha(a.raw(r)), run_manifest_sha256=a.sha(a.raw(m)),
            native_bind_proof_sha256=c.digest(p))
    snap.update(GLOBAL_sha256=a.sha(a.raw(g)), LOCAL_sha256=a.sha(a.raw(l)))
    for k, obj in [('manifest_bytes', m), ('request_bytes', r), ('global_bytes', g),
                   ('local_bytes', l), ('native_facts_bytes', p), ('snapshot_bytes', snap)]:
        v[k] = a.raw(obj)
    return v


def activated(full, tmp_path):
    """Reusable actual temp-original-onceIO helper for the store integration."""
    inputs = selected(full); proposal = d.issue(**inputs); args = {}
    for site, key, output in [('global', 'global_bytes', 'proposed_global'),
                               ('local', 'local_bytes', 'proposed_LOCAL')]:
        path = tmp_path/(site+'.json'); path.write_bytes(inputs[key])
        intent = tmp_path/(site+'.INTENT.json'); receipt = tmp_path/(site+'.RECEIPT.json')
        once.commit(path, expected_before_sha256=a.sha(inputs[key]),
            proposed_bytes=a.raw(proposal[output]), request_id='TEST-DELEGATE-'+site,
            intent_path=intent, receipt_path=receipt, lock_path=tmp_path/(site+'.lock'))
        assert once.reconcile(path, intent_path=intent, receipt_path=receipt)['status'] == \
            'CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
        args[key] = path.read_bytes(); args[site+'_intent_bytes'] = intent.read_bytes()
        args[site+'_receipt_bytes'] = receipt.read_bytes()
    grant_bytes = a.raw(proposal['grant'])
    state = d.activate(grant_bytes, **args, now_epoch=1206.)
    return dict(inputs=inputs, proposal=proposal, grant_bytes=grant_bytes,
                state_bytes=a.raw(state), activation_args=args,
                global_bytes=args['global_bytes'], local_bytes=args['local_bytes'])


@pytest.fixture
def delegated(full, tmp_path):
    return activated(full, tmp_path)


def debit(v, kind='model_loads', count=1, *, now=1207., **changes):
    state = json.loads(v['state_bytes']); p = json.loads(v['inputs']['native_facts_bytes'])
    p['observed_epoch'] = now
    args = dict(effect_id='{}:{}:{}:DEBIT'.format(state['row']['pid'],
        state['row']['proc_start'], state['sequence']), kind=kind, count=count,
        native_facts_bytes=a.raw(p), now_epoch=now)
    args.update(changes)
    out = d.consume(v['grant_bytes'], v['state_bytes'], **args)
    v['state_bytes'] = a.raw(out)
    return out


def terminal_inputs(v):
    row = json.loads(v['state_bytes'])['row']; q, loads, cpu = c.progress(row)
    p = dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_TERMINAL_V1', request_id=row['request_id'],
        native_handle=row['native_handle'], hostname=row['hostname'], uid=2009, GPU_uuid=row['GPU_uuid'],
        boot_id=row['boot_id'], pid=row['pid'], proc_start=row['proc_start'], process_state='TERMINAL',
        request_sha256=row['request_sha256'], submission_nonce=row['submission_nonce'],
        recorded_process_terminal=True, terminal_inventory_stable=True,
        exit_evidence_kind='SYSTEMD_EXEC_MAIN_EXIT', native_exit_status=0,
        finished_epoch=1400., observed_epoch=1401., conservative_GPU_s=260,
        conservative_CPU_s=max(40, cpu), new_resident_allocated_upper_bytes=row['additional_output_upper_bytes'])
    effects = dict(schema='PV1_CORE_GPU_NATIVE_DEV_EFFECT_SUMMARY_V1', request_id=row['request_id'],
        native_handle=row['native_handle'], request_sha256=row['request_sha256'],
        source_sha=row['source_sha'], run_manifest_sha256=row['run_manifest_sha256'],
        evaluation_candidate=row['evaluation_candidate'], checkpoint_manifest_sha256=row['checkpoint_manifest_sha256'],
        statistics_sha256=a.STATS, optimizer_updates_committed=0, new_physics=0, new_episodes=0,
        effect_journal_complete=True, query_attempts=q, model_load_attempts=loads, unknown_query_intents=[])
    physical = dict(schema=LEDGER_SCHEMA, run_uuid=row['child_run_uuid'], authority_release=s.TASK_SOURCE,
        Core_CPU_s=350., main_starts=1, engine_starts=3, live_and_recovery_integrations=200,
        shadow_integrations=341, reservations={}, unknown_effects=[])
    delivery = dict(status='ACTUAL_CLOSED_C50_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED',
        checkpoint_manifest_sha256=row['checkpoint_manifest_sha256'], full_resume_files_present=False,
        new_GPU_model_queries_physics=0, HF_revision='e'*40,
        files={f'model-{i}.safetensors':{} for i in range(3)})
    return dict(native_proof=p, effects=effects, physical_bytes=a.raw(physical),
                delivery_receipt_bytes=a.raw(delivery), now_epoch=1402.)


def merge(v, **changes):
    args = terminal_inputs(v); args.update(changes)
    return d.merge_terminal(v['grant_bytes'], v['state_bytes'],
                            v['global_bytes'], v['local_bytes'], **args)


def test_issue_reuses_exact_existing_reservations_and_all_original_budget_gates(full):
    inputs = selected(full); before = deepcopy(inputs); result = d.issue(**inputs)
    g, l = json.loads(inputs['global_bytes']), json.loads(inputs['local_bytes'])
    vg, vl = result['proposed_global'], result['proposed_LOCAL']
    assert inputs == before
    assert {k:v for k,v in vg.items() if k not in (d.FIELD, 'revision')} == \
        {k:v for k,v in g.items() if k != 'revision'}
    assert {k:v for k,v in vl.items() if k != d.FIELD} == l
    proof = result['grant']['body']['admission']
    assert proof['dimensions']['PBS_submissions']['matrix'] == 3
    assert proof['dimensions']['H200_s']['final'] == 72000
    assert not result['native_IO_performed'] and not result['model_permission']


@pytest.mark.parametrize('case', range(5))
def test_no_old_selector_stale_snapshot_unknown_or_budget_shortfall_grant(full, case):
    inputs = selected(full)
    if case == 0:
        inputs = full
    if case == 1:
        inputs['now_epoch'] = 1400.
    if case == 2:
        g = json.loads(inputs['global_bytes']); g['unknown_effects'] = ['unreconciled']
        inputs['global_bytes'] = a.raw(g)
    if case == 3:
        snap = json.loads(inputs['snapshot_bytes']); snap['used']['H200_s'] = 121*3600
        inputs['snapshot_bytes'] = a.raw(snap)
    if case == 4:
        snap = json.loads(inputs['snapshot_bytes']); snap['heartbeat_only_raw_ledger'] = True
        inputs['snapshot_bytes'] = a.raw(snap)
    with pytest.raises((PermissionError, KeyError)):
        d.issue(**inputs)


def test_actual_once_adoption_is_required_and_cannot_be_replayed(delegated, tmp_path):
    v = delegated; state = d.validate_state(v['grant_bytes'], v['state_bytes'])
    assert state['sequence'] == 0 and state['canonical_GLOBAL_is_fresh'] is False
    assert state['canonical_GLOBAL_observed_epoch'] == 1205.
    assert not state['model_permission'] and not state['physics_authorized']
    with pytest.raises(FileExistsError):
        once.commit(tmp_path/'global.json', expected_before_sha256=a.sha(v['inputs']['global_bytes']),
            proposed_bytes=v['global_bytes'], request_id='TEST-DELEGATE-global',
            intent_path=tmp_path/'global.INTENT.json', receipt_path=tmp_path/'global.RECEIPT.json',
            lock_path=tmp_path/'global.lock')
    bad = deepcopy(v['activation_args']); receipt = json.loads(bad['local_receipt_bytes'])
    receipt['after_sha256'] = '0'*64; bad['local_receipt_bytes'] = a.raw(receipt)
    with pytest.raises(PermissionError):
        d.activate(v['grant_bytes'], **bad, now_epoch=1206.)


def test_local_effect_chain_matches_original_charge_without_claiming_fresh_global(delegated):
    v = delegated; g = json.loads(v['global_bytes']); rid = json.loads(v['state_bytes'])['row']['request_id']
    for kind, count, now in [('model_loads', 1, 1900.), ('generation_queries', 1, 1901.), ('CPU_s', 3.75, 1902.)]:
        previous = v['state_bytes']; result = debit(v, kind, count, now=now)
        event = result['events'][-1]
        assert event['previous_state_sha256'] == a.sha(previous)
        g = c.charge(g, request_id=rid, native_handle=result['row']['native_handle'],
                     kind=kind, count=count, effect_id=event['effect_id'])
        assert g[c.FIELD][rid] == result['row']
    assert d.validate_state(v['grant_bytes'], v['state_bytes']) == result
    assert result['canonical_GLOBAL_observed_epoch'] == 1205.
    assert result['canonical_GLOBAL_is_fresh'] is False
    assert result['canonical_GLOBAL_sha256'] == a.sha(v['inputs']['global_bytes'])
    # The actual canonical input remains held and untouched by this pure local debit.
    assert c.progress(json.loads(v['global_bytes'])[c.FIELD][rid]) == (0, 0, 0)


@pytest.mark.parametrize('case', range(12))
def test_local_bound_identity_time_device_stop_and_unknown_refusals(delegated, case):
    v = delegated; p = json.loads(v['inputs']['native_facts_bytes']); p['observed_epoch'] = 1207.
    changes = {}
    if case == 0: p['pid'] += 1
    if case == 1: p['proc_start'] += 1
    if case == 2: p['source_sha'] = '0'*40
    if case == 3: p['submission_nonce'] = '00000000-0000-4000-8000-000000000000'
    if case == 4: p['native_unit_properties']['MainPID'] = '99'
    if case == 5: p['native_GPU_raw']['processes']['stdout'] = p['GPU_uuid']+', 999, unrelated, 9000\n'
    if case == 6: p['observed_epoch'] = 1000.
    if case == 7: changes['stopped'] = True
    if case == 8: changes['now_epoch'] = json.loads(v['grant_bytes'])['body']['execution_not_after_epoch']
    if case == 9:
        state = json.loads(v['state_bytes']); state['unknown_effects'] = ['lost-reply']
        v['state_bytes'] = a.raw(state)
    if case == 10:
        state = json.loads(v['state_bytes']); state['pending_effects'] = ['write-in-flight']
        v['state_bytes'] = a.raw(state)
    if case == 11: changes['effect_id'] = 'other-actor:0:0:DEBIT'
    changes['native_facts_bytes'] = a.raw(p)
    with pytest.raises(PermissionError): debit(v, **changes)


def test_duplicate_content_tokens_queries_before_load_and_resource_bounds(delegated):
    v = delegated
    with pytest.raises(PermissionError): debit(v, 'generation_queries')
    first = debit(v); token = first['events'][0]['effect_id']
    with pytest.raises(PermissionError): debit(v, 'generation_queries', effect_id=token)
    with pytest.raises(PermissionError): debit(v, 'CPU_s', first['row']['CPU_upper_s']+1)
    with pytest.raises(PermissionError): debit(v, 'model_loads', True)
    with pytest.raises(PermissionError): debit(v, 'optimizer_updates', 1)
    debit(v)
    with pytest.raises(PermissionError): debit(v)


def test_runtime_own_model_memory_is_allowed_but_not_a_new_foreign_process(delegated):
    from scripts.pi0294.pv1.evaluation_core_identity import parse_GPU_rows
    v = delegated; debit(v)
    p = json.loads(v['inputs']['native_facts_bytes']); p['observed_epoch'] = 1900.
    # A loaded model consumes memory. This is a runtime identity read, not
    # another requirement to have the original premodel free-memory amount.
    parts = p['native_GPU_raw']['devices']['stdout'].strip().split(',')
    parts[6] = '16000'; p['native_GPU_raw']['devices']['stdout'] = ','.join(parts)+'\n'
    own = p['GPU_uuid']+', '+str(p['pid'])+', python-neural, 15000\n'
    p['native_GPU_raw']['processes']['stdout'] = own
    p['device'] = parse_GPU_rows(p['native_GPU_raw']['devices']['stdout'], own,
        GPU_uuid=p['GPU_uuid'], own_pid=p['pid'])
    assert debit(v, 'generation_queries', now=1900., native_facts_bytes=a.raw(p))['sequence'] == 2
    p['native_GPU_raw']['processes']['stdout'] += p['GPU_uuid']+', 999, unrelated, 100\n'
    with pytest.raises(PermissionError, match='FOREIGN_COMPUTE'):
        debit(v, 'generation_queries', now=1900., native_facts_bytes=a.raw(p))


@pytest.mark.parametrize('case', range(7))
def test_hash_chain_counter_or_global_freshness_tampering_is_rejected(delegated, case):
    v = delegated; debit(v); state = json.loads(v['state_bytes'])
    if case == 0: state['row']['charged_model_loads'] = 0
    if case == 1: state['canonical_GLOBAL_is_fresh'] = True
    if case == 2: state['events'][0]['previous_state_sha256'] = '0'*64
    if case == 3: state['events'][0]['count'] = 2
    if case == 4:
        grant = json.loads(v['grant_bytes']); grant['body']['limits']['CPU_s'] += 1
        v['grant_bytes'] = a.raw(grant)
    if case == 5: state['event_chain_head_sha256'] = '0'*64
    if case == 6: state['events'][0]['previous_event_sha256'] = '0'*64
    with pytest.raises(PermissionError): d.validate_state(v['grant_bytes'], a.raw(state))


def test_remote_writer_fence_preserves_old_path_and_rejects_delegated_path(delegated):
    g = json.loads(delegated['global_bytes']); rid = next(iter(g[d.FIELD]))
    assert d.assert_remote_writer_allowed(json.loads(delegated['inputs']['global_bytes']), rid)
    with pytest.raises(PermissionError): d.assert_remote_writer_allowed(g, rid)


def test_terminal_merges_only_own_costs_with_original_settlement_and_preserves_new_other_scope_changes(delegated):
    v = delegated; debit(v); debit(v, 'generation_queries', now=1208.); debit(v, 'CPU_s', 12.5, now=1209.)
    g, l = json.loads(v['global_bytes']), json.loads(v['local_bytes'])
    g['other_scope_actual_evidence'] = dict(cost=23, immutable='preserve')
    l['other_scope_actual_evidence'] = dict(cost=23, immutable='preserve')
    g['CPU_s'] += 23; g['revision'] += 7
    v.update(global_bytes=a.raw(g), local_bytes=a.raw(l)); before = deepcopy(v)
    expected = deepcopy(g); state = json.loads(v['state_bytes']); rid = state['row']['request_id']
    for e in state['events']:
        expected = c.charge(expected, request_id=rid, native_handle=state['row']['native_handle'],
                            kind=e['kind'], count=e['count'], effect_id=e['effect_id'])
    args = terminal_inputs(v)
    original = terminal.propose(a.raw(expected), a.raw(l), request_id=rid, **args)
    result = merge(v); vg, vl = result['proposed_global'], result['proposed_LOCAL']
    assert v == before
    assert {k:x for k,x in vg.items() if k != d.FIELD} == \
        {k:x for k,x in original['proposed_global'].items() if k != d.FIELD}
    assert {k:x for k,x in vl.items() if k != d.FIELD} == \
        {k:x for k,x in original['proposed_LOCAL'].items() if k != d.FIELD}
    assert vg['other_scope_actual_evidence'] == g['other_scope_actual_evidence']
    assert vg['CPU_s'] == g['CPU_s'] + 40
    assert vg['used']['H200_s'] == g['used']['H200_s'] + 260
    assert vg[d.FIELD][rid]['status'] == d.MERGED
    assert vg[d.FIELD][rid]['original_GLOBAL_sha256'] == a.sha(v['inputs']['global_bytes'])
    assert not result['leases_closed'] and not result['scientific_stage_advanced']
    v.update(global_bytes=a.raw(vg), local_bytes=a.raw(vl))
    with pytest.raises(PermissionError): merge(v)


@pytest.mark.parametrize('case', range(6))
def test_terminal_rejects_other_writer_unknown_mismatch_and_unproven_exit(delegated, case):
    v = delegated; debit(v); changes = {}
    if case == 0:
        g = json.loads(v['global_bytes']); state = json.loads(v['state_bytes']); row = state['row']
        g = c.charge(g, request_id=row['request_id'], native_handle=row['native_handle'],
                     kind='CPU_s', count=1, effect_id='other/CPU')
        v['global_bytes'] = a.raw(g)
    if case == 1:
        g = json.loads(v['global_bytes']); g['unknown_effects'] = ['unreconciled']
        v['global_bytes'] = a.raw(g)
    args = terminal_inputs(v)
    if case == 2: args['native_proof']['recorded_process_terminal'] = False
    if case == 3: args['effects']['query_attempts'] = 1
    if case == 4: args['effects']['unknown_query_intents'] = [0]
    if case == 5: args['native_proof']['finished_epoch'] = 1206.
    changes.update(args)
    with pytest.raises((PermissionError, KeyError)): merge(v, **changes)
