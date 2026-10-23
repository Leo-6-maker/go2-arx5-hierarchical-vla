"""Shared pure recovery2 terminal contracts; caller pins the exact native identity.

No transport, file access or native dispatch. These predicates never turn
CPU fixtures into completion, and retain the original full saved-state audit.
"""
import base64
import hashlib
import json
import re
import uuid
from pathlib import PurePosixPath
from . import b_vanda_recovery2 as recovery,b_heartbeat_recovery as b,b_saved_state_reuse as saved
from .b_recovery_completion import identity
from . import a40_allocation
from scripts.pi0294.e1.journal import digest

def need(ok,why):
    if not ok:raise PermissionError('B_VANDA_RECOVERY2_COMPLETION_'+why)

class CompletionContract:
    def __init__(self, *, source, run, authority=recovery):
        need(re.fullmatch('[0-9a-f]{40}',source) is not None and isinstance(run,str) and run,
             'EXPLICIT_PINNED_AUDITOR_SOURCE_AND_RUN')
        self.source=source
        self.run=run
        self.authority=authority

    def expected(self, value):
        r = value['request']
        encoded = base64.b64decode(value['request_bytes_base64'], validate=True)
        need(json.loads(encoded) == r and hashlib.sha256(encoded).hexdigest() == value['request_sha256'], 'ACTUAL_REQUEST_BYTES_NOT_REBUILT_OR_CHANGED')
        try:
            valid_ids = all((str(uuid.UUID(r[k])) == r[k] for k in ('request_id', 'submission_nonce')))
        except (ValueError, TypeError, KeyError, AttributeError):
            valid_ids = False
        reg = value['registration']
        row = value['bound_submission']
        bound = value['bound_ledger']
        need(valid_ids and re.fullmatch('[0-9]+\\.stdct-mgmt-02', value['job_id']) is not None and (value['job_id'] not in (recovery.JOB, saved.JOB, b.FAILED_JOB)) and (r.get('source_sha') == reg.get('source_sha') == self.source) and (r.get('run_uuid') == self.run) and (PurePosixPath(reg['root']).name == self.run) and (r.get('run_manifest_sha256') == reg.get('manifest_sha256')) and (re.fullmatch('[0-9a-f]{64}', reg.get('manifest_sha256', '')) is not None) and (r.get('expected_native_start_step') == 2048) and (r.get('training_block', {}).get('stop_after_step') == 4096) and (r.get('CPU_upper_s') == 54000) and (r.get('additional_output_upper_bytes') == 70000000000) and self.authority.request(r), 'NEW_REGISTERED_SAME_B2048_TO4096_RECOVERY2_ONLY')
        wanted = dict(job_id=value['job_id'], request_id=r['request_id'], request_sha256=value['request_sha256'], submission_nonce=r['submission_nonce'], source_sha=self.source, training_lineage_key=recovery.LINEAGE, start_native_step=2048, segment_stop_step=4096, status='SUBMITTED_PENDING_NATIVE', optimizer_update_limit=2048, query_limit=29)
        need(all((row.get(k) == v for (k, v) in wanted.items())), 'EXACT_IMMUTABLE_NATIVE_BOUND_REPLY')
        self.authority.validate_global(bound)
        matches = [v for v in bound['PBS_submissions'] if v.get('request_id') == r['request_id']]
        need(matches == [row] and bound['optimizer_updates'] == 9666 and (bound['generation_queries'] == 1214) and (bound['training_runs'][recovery.LINEAGE]['last_native_step'] == 2048), 'BOUND_BASELINE9666_WITH_OLD2970_AND_ALL_PAID_LOSSES_PRESERVED')
        return r

    def owner(self, packet, value, *, now_epoch):
        self.expected(value)
        exit_record = packet['OS_exit']
        process = packet['owner_process']
        launch = packet['launch']
        current = packet['current_process']
        need(type(exit_record.get('returncode')) is int and exit_record['returncode'] == 0 and (exit_record.get('schema') == 'VANDA_ORIGINAL_FOREGROUND_OWNER_OS_EXIT_V1') and (exit_record.get('foreground_wait_completed') is True) and (exit_record.get('source_sha') == self.source) and (exit_record.get('owner_process') == process) and (exit_record.get('wrapper_pid') == launch.get('pid')) and (launch.get('process_kind') == 'ORIGINAL_FOREGROUND_EXIT_CAPTURE_WRAPPER') and (launch.get('source_sha') == process.get('source_sha') == self.source) and (type(launch.get('pid')) is int) and (launch['pid'] > 1) and (process.get('uid') == 22182) and (type(process.get('pid')) is int) and (process['pid'] > 1) and (re.fullmatch('[0-9]+', process.get('proc_start_ticks', '')) is not None) and (type(process.get('maximum_qsub')) is int) and (process['maximum_qsub'] == 1) and (process.get('agent_connection_required') is False) and (packet.get('preserved_stop_exists') is False), 'REAL_FOREGROUND_OWNER_OS_EXIT0_NOT_ONLY_AN_EXIT_FILE')
        need(current.get('pid') == process['pid'] and current.get('present_same_instance') is False and (current.get('physical_platform') == 'VANDA_A40') and (type(current.get('observed_epoch')) in (int, float)) and (0 <= now_epoch - current['observed_epoch'] <= 120) and (current.get('proc_start_ticks') is None or current['proc_start_ticks'] != process['proc_start_ticks']), 'FRESH_EXACT_OWNER_PROCESS_ABSENT_OR_PID_REUSED_NOT_TIMEOUT')

    def native(self, job, value):
        self.expected(value)
        identity(job, value)
        need(job.get('job_state') == 'F' and type(job.get('Exit_status')) is int and (job['Exit_status'] == 0), 'CURRENT_MATCHING_NATIVE_F_EXIT0')

    def effects(self, e, value):
        r = self.expected(value)
        wanted = dict(job_id=value['job_id'], request_id=r['request_id'], request_sha256=value['request_sha256'], source_sha=self.source, arm='B_COVERAGE', start_native_step=2048, last_native_step=4096, optimizer_updates_committed=2048, recipe_sha256=b.RECIPE, pending_updates=[], unknown_query_intents=[], effect_journal_complete=True, new_physics=0, new_episodes=0)
        need(all((e.get(k) == v for (k, v) in wanted.items())) and all((type(e.get(k)) is int for k in ('start_native_step', 'last_native_step', 'optimizer_updates_committed', 'new_physics', 'new_episodes'))) and (type(e.get('query_attempts')) is int) and (17 <= e['query_attempts'] <= 29) and (type(e.get('model_load_attempts')) is int) and (1 <= e['model_load_attempts'] <= 4), 'COMPLETE_ACTUAL_EFFECTS_NOT_RESERVED_QUERY_OR_LOAD_COUNTS')

    def origin(self, step, complete, value):
        self.expected(value)
        origins = {1024: (b.FAILED_JOB, b.FAILED_SOURCE), 2048: (saved.JOB, saved.SOURCE), 4096: (value['job_id'], self.source)}
        need(step in origins and complete.get('global_step') == step and (complete.get('full_resume') is True) and (len(complete.get('files', {})) == 15) and ((complete.get('origin_job_id'), complete.get('origin_source_sha')) == origins[step]), 'SEPARATE_FAILED1024_SAVED2048_AND_NEW4096_ORIGINS')

    def query_provenance(self, v, value):
        """Join actual journal bytes to all original 17 compatibility arrays.

        Financial effects retain their original bounds. Successful source90 has
        exactly these draws; replaying saved probes cannot become completion.
        The independent reader must hash the files, not merely echo the receipt.
        """
        self.effects(v['effects'], value)
        ledger = json.loads(base64.b64decode(v['native_effect_ledger_base64'], validate=True))
        result = json.loads(base64.b64decode(v['compatibility_result_base64'], validate=True))
        need(ledger.get('job_id') == result.get('job_id') == value['job_id'] and ledger.get('request_sha256') == result.get('request_sha256') == value['request_sha256'] and (ledger.get('source_sha') == self.source) and (ledger.get('run_uuid') == self.run) and (ledger.get('generation_queries') == v['effects']['query_attempts'] == 17) and (ledger.get('model_loads') == v['effects']['model_load_attempts']) and (ledger.get('optimizer_updates') == 4096) and (result.get('status') == 'PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS') and (result.get('parent_checkpoint_manifest_sha256') == a40_allocation.PARENT) and (result.get('statistics_sha256') == a40_allocation.R0) and all((type(result.get(k)) is int and result[k] == 0 for k in ('new_physics', 'new_tasks', 'new_optimizer_updates'))), 'ACTUAL_SAME_JOB_COMPATIBILITY_NOT_REPLAYED_PROBES')
        links = ledger['query_intent_links']
        records = result['records']
        rows = v['query_intent_bytes_base64']
        need(len(links) == len(records) == len(rows) == len(v['compatibility_array_sha256']) == 17 and set(rows) == {x['intent_relative'] for x in links}, 'FULL_QUERY_CENSUS_NOT_SELECTED_SUBSET')
        for (i, (link, record)) in enumerate(zip(links, records)):
            message = 'probe-A40-COMPATIBILITY-draw-' + str(i)
            path = 'reservations/probe-generations/' + message + '.json'
            need(link.get('intent_relative') == path and path in rows, 'EXACT_COMPATIBILITY_INTENT_PATH')
            row = json.loads(base64.b64decode(rows[path], validate=True))
            array = v['compatibility_array_sha256'].get('A40_COMPATIBILITY/query-' + str(i).zfill(2) + '.npz')
            need(link.get('index') == record.get('index') == i and record.get('seed') == 29400 + i and (record.get('finite_shapes_verified') is True) and (link.get('message_id') == row.get('message_id') == message) and (link.get('intent_relative') == path) and (row.get('run_uuid') == self.run) and (row.get('status') == 'COMPLETE') and (link.get('payload_sha256') == row.get('payload_sha256')) and (re.fullmatch('[0-9a-f]{64}', row.get('payload_sha256', '')) is not None) and (row.get('result_sha256') == digest(row.get('result'))) and (re.fullmatch('[0-9a-f]{64}', array or '') is not None) and (row['result'].get('arrays_sha256') == record.get('arrays_sha256') == array) and (row['result'].get('inference_precision') == record.get('inference_precision')), 'EXACT_COMPLETED_COMPATIBILITY_INTENT_AND_FULL_ARRAY_READBACK')

    def audit(self, v, value):
        r = self.expected(value)
        self.native(v['raw_native'], value)
        self.effects(v['effects'], value)
        self.query_provenance(v, value)
        (f, p, w) = (v['process_final'], v['native_proof'], v['wrapper_final'])
        job = value['job_id']
        need(p.get('job_id') == job and p.get('request_id') == r['request_id'] and (p.get('job_state') == 'F') and (type(p.get('Exit_status')) is int) and (p['Exit_status'] == 0) and (p.get('recorded_process_terminal') is True) and (p.get('new_resident_allocated_upper_bytes') == r['additional_output_upper_bytes']), 'REAL_TERMINAL_AND_FULL_ORIGINAL_OUTPUT_BOUND')
        need(f.get('status') == 'A40_SOURCE_4096_COMPLETE' and f.get('global_step') == f.get('last_returned_training_step') == 4096 and (f.get('error') is None) and (f.get('request_sha256') == value['request_sha256']) and (f.get('source_sha') == self.source) and (f.get('job_id') == w.get('job_id') == job) and (type(w.get('returncode')) is int) and (w['returncode'] == 0) and ({k: z for (k, z) in f.get('effects', {}).items() if k != 'finished_epoch'} == {k: z for (k, z) in v['effects'].items() if k != 'finished_epoch'}), 'EXACT_RETURNED4096_AND_SAME_ACTUAL_EFFECTS')
        need(set(v['checkpoints']) == set(v['checkpoint_members']) == {'1024', '2048', '4096'} and v.get('probe_count') == 12 and (len(v.get('probe_files', {})) == 13) and (v.get('new_1024_probe_queries') == 0) and (v.get('imported1024_and2048_origin_preserved') is True) and all((v.get(k) == 0 for k in ('models_loaded', 'new_queries', 'new_updates', 'new_physics'))), 'THREE_FULL_STATES_AND_ORIGINAL_PROBES_READONLY')
        required = {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'SAMPLER_STATE.json', 'trainer_state.json', 'statistics.json', 'config.json'}
        for step in (1024, 2048, 4096):
            row = v['checkpoints'][str(step)]
            members = v['checkpoint_members'][str(step)]
            need(row.get('members') == len(members) == 15 and row.get('actual_consumed_cursor') == step * 32 and (required <= set(members)) and any((n.startswith('processor/') for n in members)) and any((n.endswith('.safetensors') for n in members)) and all((re.fullmatch('[0-9a-f]{64}', s) for s in members.values())) and (re.fullmatch('[0-9a-f]{64}', row.get('complete_sha256', '')) is not None), 'ALL_OPTIMIZER_SCHEDULER_RNG_SAMPLER_PROCESSOR_STATISTICS_AND_MODEL_MEMBERS')
        need(v['checkpoints']['1024']['complete_sha256'] == b.SEED_SHA and v['checkpoints']['2048']['complete_sha256'] == saved.CHECKPOINT and (v.get('probe_result_sha256') == b.PROBE_SHA), 'UNCHANGED_ORIGINAL_SAVED_STATES_AND12_PROBES')
        probe = base64.b64decode(v['original_probe_result_base64'], validate=True)
        need(hashlib.sha256(probe).hexdigest() == b.PROBE_SHA and v['probe_files'].get('evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json') == b.PROBE_SHA and all((n.startswith('evidence/B_COVERAGE-1024/TRAIN_PROBES/') and re.fullmatch('[0-9a-f]{64}', s) for (n, s) in v['probe_files'].items())), 'ACTUAL_ORIGINAL_PROBE_RESULT_BYTES_AND_ARRAY_HASHES')
        return v

    def settled(self, v, value, actual_effects):
        r = self.expected(value)
        self.effects(actual_effects, value)
        (h, d) = (v['ledger'], v['DEC1'])
        self.authority.validate_adopted(h, d)
        before = value['bound_ledger']
        queries = actual_effects['query_attempts']
        loads = actual_effects['model_load_attempts']
        need(v['job_id'] == value['job_id'] and v['request_id'] == r['request_id'] and (v['updates'] == h['optimizer_updates'] == d['optimizer_updates'] == 11714) and (v['queries'] == h['generation_queries'] == before['generation_queries'] + queries) and (h['model_loads'] == before['model_loads'] + loads) and (v['unknown_effects'] == h['unknown_effects'] == h['pending_updates'] == d['unknown_effects'] == []), 'ACTUAL9666_PLUS2048_AND_ACTUAL_QUERY_LOAD_DELTAS')
        need(h['training_runs'][recovery.LINEAGE]['last_native_step'] == 4096 and h['training_runs'][b.LINEAGE]['last_native_step'] == 2970 and (h['training_runs']['B_COVERAGE']['last_native_step'] == 1829) and (h['training_runs'][b.zero.HR.LINEAGE]['last_native_step'] == 4096), 'NEW_PROGRESS_WITH_PAID922_AND805_LOSSES_AND_A4096_PRESERVED')
        rows = [x for x in h['PBS_submissions'] if x.get('request_id') == r['request_id']]
        need(len(rows) == 1 and rows[0]['job_id'] == value['job_id'] and (rows[0]['source_sha'] == self.source) and (rows[0]['request_sha256'] == value['request_sha256']) and (rows[0]['status'] == 'NATIVE_TERMINAL') and (rows[0]['native_exit_status'] == 0) and (rows[0]['committed_optimizer_updates'] == 2048) and (rows[0]['query_attempts'] == queries) and (rows[0]['reserved_optimizer_updates'] == rows[0]['reserved_neural_queries'] == 0), 'ONE_MATCHING_SETTLED_EFFECT_ROW')
        need(not any(h['reserved'].values()) and h['CPU_reserved_s'] == h['storage_reserved_bytes'] == 0 and all((d.get(k, 0) == 0 for k in ('GPU_reserved_s', 'CPU_reserved_s', 'optimizer_updates_reserved', 'neural_queries_reserved', 'GPU_site_output_reserved_bytes', 'network_transfer_reserved_bytes'))), 'ALL_ORIGINAL_RESERVATIONS_RELEASED_WITHOUT_UNKNOWN')
        return rows[0]

