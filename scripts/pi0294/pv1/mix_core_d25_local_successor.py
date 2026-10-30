"""Exact third D25 panel: retained paid IO and a new finite control slice.

Pure accounting/metadata only. The first two failures and all their invoices
remain immutable. This selector is neither C50 credit nor first512 credit.
Original onceIO, native registration, physical guard and model gates remain
mandatory. A transport plan is a hard cap, not evidence of a completed send.
"""
from copy import deepcopy
import base64
import gzip
import json
from pathlib import PurePosixPath
import re

from . import mix_accounting as a, successor_cd as s

FIELD = 'V5_CD_D25_exact_e686_local_successor'
SCHEMA = 'V5_CD_D25_EXACT_ZERO_LOCAL_SUCCESSOR_V1'
FIRST = '9dd25c79-9a67-482a-bce7-d88a0fc922bc'
FAILED = 'e686f017-743d-4286-8289-c8b594dfb224'
SOURCE = '9e399b397b975456147d1efc370b609ab4e3c647'
RUN = 'v5-cd-d25-n294101-03'
CHECKPOINT = '8cf977b964411a3ae97f022edd3e3ca390cd2d71fd61a914ab21466efa3a9e9b'
REVISION = '4887843bbff8f69c8c439d5456469f173194c317'
DELIVERY = 'cc2228779fcfff332e77208a9d6c2f614db249288ae7cb7a824875651d90f57d'
MiB = 1024 ** 2
GiB = 1024 ** 3
CONTROL = dict(Hopper_CPU_s=1800, Core_CPU_s=9000, local_CPU_s=1800,
               Hopper_storage_bytes=64*MiB, Core_storage_bytes=128*MiB,
               local_storage_bytes=64*MiB, transfer_bytes=64*MiB)
WIRE = dict(two_source_install_envelopes_total=32*MiB,
            fresh_canonical_capture_and_adoption_total=8*MiB,
            grant_bind_and_fixed_configuration=4*MiB,
            terminal_reconcile_merge_payload=8*MiB,
            bounded_unknown_readonly_differential=4*MiB,
            protected_closeout=8*MiB)
NINE = ('config.json', 'model-00001-of-00003.safetensors',
        'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors',
        'model.safetensors.index.json', 'processor/embodiment_id.json',
        'processor/processor_config.json', 'processor/statistics.json', 'statistics.json')


def need(ok, why):
    if not ok:
        raise PermissionError('V5_D25_LOCAL_SUCCESSOR_' + why)


def digest(v):
    return a.sha(a.raw(v))


def _inventory(value):
    need(value['schema'] == 'D25_CLOSED_ROOTS_METADATA_ONLY_INVENTORY_V1'
         and value['tensor_streams'] == value['remote_mutations'] == 0,
         'ACTUAL_METADATA_CAPTURE_NO_EFFECT')
    raw = gzip.decompress(base64.b64decode(value['inventory_gzip_base64']))
    need(a.sha(raw) == value['inventory_sha256'], 'COMPLETE_INVENTORY_BYTES')
    rows = json.loads(raw); groups = {}
    need(len(rows) == value['entries'] and all(v['exists'] for v in value['roots'].values()),
         'ALL_OLD_ROOTS_EXIST_NOT_DELETED')
    roots = [v['path'] for v in value['roots'].values()]
    need(len(roots) == len(set(roots)), 'DISTINCT_REGISTERED_ROOTS')
    for path, item in rows.items():
        need(any(path == r or path.startswith(r+'/') for r in roots)
             and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
             'EXACT_ROOT_DESCENDANT')
        need(item['type'] in ('directory', 'file', 'symlink') and item['uid'] == value['uid']
             and all(type(item[k]) is int and item[k] >= 0 for k in
                     ('dev', 'ino', 'bytes', 'blocks', 'mtime_ns', 'ctime_ns', 'nlink')),
             'OWNED_REGULAR_DIRECTORY_OR_PINNED_LINK')
        key = (item['dev'], item['ino'])
        if key in groups:
            need(groups[key][0][1] == item, 'SAME_INODE_SAME_STAT')
        groups.setdefault(key, []).append((path, item))
    for same in groups.values():
        item = same[0][1]
        if item['type'] == 'file':
            need(item['nlink'] == len(same), 'NO_UNCAPTURED_HARDLINK_CONSUMER')
    allocated = sum(x[0][1]['blocks']*512 for x in groups.values())
    total = sum(max(x[0][1]['bytes'], x[0][1]['blocks']*512) for x in groups.values())
    need(len(groups) == value['unique_inodes'] and allocated == value['unique_allocated_bytes'],
         'RECOMPUTED_UNIQUE_ALLOCATED_INCLUDING_DIRECTORIES')
    return rows, total


def inventory_guard(capture, historical_model, *, now_epoch):
    """Join complete old roots to a saved prior model SHA/stat witness."""
    core, vanda = (capture[k]['value'] for k in ('core', 'vanda'))
    for site in (core, vanda):
        need(0 <= now_epoch-site['observed_epoch'] <= 120, 'FRESH_PHYSICAL_READBACK')
    need(core['host'] == 'corelab-4080s' and core['uid'] == 2009
         and vanda['host'] == 'stdct-login-01' and vanda['uid'] == 22182,
         'ACTUAL_BOTH_HOSTS')
    need(set(core['roots']) == {'owner01','physical01','neural01','owner02','physical02',
                               'neural02','source01','source02','delivery'}
         and len(vanda['roots']) == 1, 'ALL_EXACT_PREDECESSOR_AND_DELIVERY_ROOTS')
    rows, total = _inventory(core); site_rows, site_total = _inventory(vanda)
    model = core['roots']['delivery']['path']+'/model_view'
    old = historical_model
    need(old['status'] == 'ACTUAL_D25_RESTORED_METADATA_AND_CLOSED_TENSOR_WITNESS_READONLY'
         and old['checkpoint_manifest_sha256'] == CHECKPOINT and old['revision'] == REVISION
         and set(old['files']) == set(NINE), 'ORIGINAL_NINE_FILE_DELIVERY_WITNESS')
    model_alloc = 0; model_stats = {}
    for name in NINE:
        current, prior = rows[model+'/'+name], old['files'][name]
        need(current['type'] == 'file' and current['mode'] == 0o400
             and all(current[k] == prior[j] for k,j in
                     (('dev','dev'),('ino','inode'),('bytes','bytes'),
                      ('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')))
             and re.fullmatch('[0-9a-f]{64}', prior['expected_sha256']),
             'UNCHANGED_SAVED_STREAM_IDENTITY:' + name)
        if not name.endswith('.safetensors'):
            need(prior['actual_sha256'] == prior['expected_sha256'], 'RESTORED_METADATA_SHA')
        model_alloc += max(current['bytes'], current['blocks']*512)
        model_stats[name] = dict(stat=deepcopy(current), sha256=prior['expected_sha256'])
    tensors = {p for p in rows if p.endswith('.safetensors')}
    need(tensors == {model+'/'+n for n in NINE if n.endswith('.safetensors')}
         and not any(p.endswith('.safetensors') for p in site_rows), 'NO_DUPLICATE_TENSOR_PAYLOAD')
    links = {p:v for p,v in rows.items() if v['type'] == 'symlink'}
    expected = {}
    for suffix in ('01','02'):
        root = core['roots']['neural'+suffix]['path']
        expected[root+'/model_views/D25'] = model
        cosmos = root+'/model_inputs/nvidia/Cosmos-Reason2-2B'
        need(cosmos in links and links[cosmos]['canonical_target'].endswith(
            '/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'),
            'ORIGINAL_PINNED_COSMOS_INPUT_ONLY')
        expected[cosmos] = links[cosmos]['canonical_target']
    need(set(links) == set(expected) and all(v['link_text'] == v['canonical_target'] == expected[p]
         for p,v in links.items()) and not any(v['type']=='symlink' for v in site_rows.values()),
         'ONLY_ORIGINAL_DECLARED_INPUT_SYMLINKS')
    small = core['small_files'][core['roots']['delivery']['path']+'/ACTUAL_CORE_MODEL_DELIVERED.json']
    # The capture carries actual small bytes. Different capture tools use value
    # or base64; its exact raw SHA is still required, never a caller PASS flag.
    need(small['sha256'] == DELIVERY, 'UNCHANGED_ACTUAL_DELIVERY_RECEIPT')
    need(total-model_alloc <= 2*128*MiB and site_total <= 2*64*MiB,
         'ALL_OLD_METADATA_INSIDE_TWO_ALREADY_PAID_CONTROL_UPPERS')
    return dict(schema='D25_TWO_FAILED_ROOTS_FIXED_MODEL_PHYSICAL_JOIN_V1',
        Core_inventory_sha256=core['inventory_sha256'], Vanda_inventory_sha256=vanda['inventory_sha256'],
        Core_unique_allocated_bytes=core['unique_allocated_bytes'], Core_conservative_retained_bytes=total,
        Core_model_allocated_bytes=model_alloc,
        Core_control_metadata_allocated_bytes=total-model_alloc,
        Vanda_control_metadata_allocated_bytes=site_total,
        old_Core_control_paid_bytes=2*128*MiB, old_site_control_paid_bytes=2*64*MiB,
        model_root=model, model_stats=model_stats, roots=deepcopy(core['roots']),
        Vanda_roots=deepcopy(vanda['roots']), all_old_metadata_retained=True,
        new_tensor_transfers=0, new_tensor_streams=0)


def _closed(g, l, closed, proof):
    from . import mix_core_accounting as c
    from .evaluation_core_accounting import terminal_values
    row = g[c.FIELD][FAILED]
    need(row['source_sha'] == SOURCE and row['status'] == c.STATES[3]
         and row['native_exit_status'] == 1 and row['evaluation_candidate'] == 'D25'
         and row['checkpoint_manifest_sha256'] == CHECKPOINT
         and row['query_attempts'] == row['model_load_attempts'] == 0
         and row['unknown_query_intents'] == [], 'EXACT_KNOWN_ZERO_PREDECESSOR')
    native, effects = proof['native_proof'], proof['effects']
    values = terminal_values(row, request_id=FAILED, native_proof=native,
                             effects=effects, now_epoch=row['settled_epoch'])
    need(values['gpu'] == 656 and values['cpu'] == 4 and values['queries'] == values['loads'] == 0
         and not values['unknown'] and c.digest(native) == row['terminal_native_sha256']
         and c.digest(effects) == row['terminal_effect_summary_sha256'], 'FULL_ORIGINAL_TERMINAL_VALIDATION')
    for rid in (FIRST, FAILED):
        r, receipt = g[c.FIELD][rid], l['V5_CD_Core_terminal_settlements'][rid]
        need(r['status'] == c.STATES[3] and r['query_attempts'] == r['model_load_attempts'] == 0
             and receipt['request_id'] == rid and receipt['terminal_native_sha256'] == r['terminal_native_sha256']
             and receipt['same_panel_GPU_output_upper_already_paid'] == 0
             and receipt['actual_model_delivery_receipt_sha256'] == DELIVERY
             and all(receipt['physical'][k] == 0 for k in ('episodes','constructors','live','shadow')),
             'BOTH_ZERO_FAILS_AND_ALL_GROSS_INVOICES_RETAINED')
        for field, key in (('Core_output_charged_upper_bytes','Core_storage_bytes'),
                           ('transfer_charged_upper_bytes','transfer_bytes'),
                           ('original_gross_GPU_output_upper_bytes','Hopper_storage_bytes')):
            need(receipt[field] == c.remaining_stage()[key], 'ORIGINAL_FULL_GROSS_PAID')
    need(closed['request_id'] == FAILED and set(closed['leases']) == {'GPU','physical','control'}
         and all(closed[k] == 0 for k in ('valid_conditions','model_loads','queries','new_physics','new_writes')),
         'INDEPENDENT_ZERO_HISTORY_THREE_CLOSED')
    for role, item in closed['leases'].items():
        lease = item['value']; reconcile = item['reconcile']
        need(digest(lease) == item['sha256'] and lease['status'] == 'CLOSED'
             and lease['terminal_request_id'] == FAILED and lease['all_actual_workers_terminal'] is True
             and lease['terminal_native_sha256'] == row['terminal_native_sha256']
             and reconcile['status'] == 'CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
             and reconcile['actual_sha256'] == item['sha256']
             and reconcile['receipt_binding_valid'] is True,
             'ORIGINAL_ONCE_CLOSED_' + role)
    need(len(closed['units']) == 2 and all(v['MainPID'] == '0' and v['ExecMainStatus'] == '1'
         and v['ActiveState'] == 'failed' for v in closed['units'].values()), 'REAL_BOTH_OS_TERMINAL')


def _LOCAL_paid(l,m):
    """Actual Core table, never a fabricated scalar GLOBAL projection."""
    from .evaluation_scope import CAMPAIGN
    b=m['financial_before']
    need(l.get('run_uuid')==CAMPAIGN and type(l.get('CPU_s'))is dict
         and set(l['CPU_s'])=={'Hopper','CoreLab','local_PC'},'ACTUAL_LOCAL_CAMPAIGN_AND_THREE_SITE_CPU_TABLE')
    for name,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:
        need(type(l['CPU_s'][site])in(int,float)
             and l['CPU_s'][site]>=b['local_CPU_s'][site]+CONTROL[name],'NEW_CONTROL_PAID_'+site)
    need(l['network_transfer_accounted_bytes']>=b['network_transfer_accounted_bytes']+CONTROL['transfer_bytes']
         and l['Core_native_GPU_current_storage_bytes']>=b['Core_native_GPU_current_storage_bytes']+CONTROL['Core_storage_bytes']
         and l.get('V5_CD_finite_control_local_storage_bytes',{}).get(m['replacement_request_id'])==CONTROL['local_storage_bytes'],
         'COMPLETE_NEW_CONTROL_RETAINED_IO_NOT_OLD_PREPAY')


def validated(g):
    """Nonrecursive marker validator; safe inside original row validation."""
    m = g.get(FIELD)
    if m is None:
        return None
    from . import mix_core_accounting as c, mix_core_d25_continuation as prior
    need(m['schema'] == SCHEMA and m['failed_request_id'] == FAILED and m['failed_source_sha'] == SOURCE
         and m['scope_grant_sha256'] == s.GRANT_SHA and m['T_CD'] == s.validated(g)['T_CD']
         and m['deadline_epoch'] == s.validated(g)['deadline_epoch'], 'UNCHANGED_SCOPE_CLOCK')
    need(m['prior_marker_sha256'] == digest(g[prior.FIELD])
         and m['old_prepaid_control'] == g[a.PACKAGE]['prepaid_control'][c.key('D25')]
         and m['engineering_replacements_used'] == g[a.PACKAGE]['engineering_replacements_used'] == 4,
         'UNCHANGED_OLD_MARKERS_CONTROL_AND_LIMIT')
    for rid in (FIRST, FAILED):
        need(digest(g[c.FIELD][rid]) == m['failed_row_sha256'][rid], 'OLD_FAILED_ROW_BYTES_RETAINED')
    need(m['paid_IO_credit'] == {k:c.remaining_stage()[k] for k in
         ('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')}
         and m['finite_control_costs'] == CONTROL and m['wire_phase_upper_bytes'] == WIRE
         and m['financial_control_paid'] is True and m['maximum_new_requests'] == 1,
         'EXACT_ONE_PANEL_AND_FULL_NEW_CONTROL_COSTS')
    need(c.canonical_uuid(m['replacement_request_id']) and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id'] not in (FIRST,FAILED) and m['replacement_run_uuid'] == RUN
         and re.fullmatch('[0-9a-f]{40}', m['source_sha']), 'DISTINCT_COLD_IDENTITY')
    if type(g.get('CPU_s'))is dict:
        _LOCAL_paid(g,m)
    else:
        need(g.get('schema')=='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
             and type(g.get('CPU_s'))in(int,float)
             and g['CPU_s'] >= m['financial_before']['site_CPU_s']+CONTROL['Hopper_CPU_s']
             and g['new_storage_current_bytes'] >= m['financial_before']['site_current_bytes']+CONTROL['Hopper_storage_bytes']
             and g['new_storage_peak_bytes'] >= m['financial_before']['site_peak_bytes'], 'NO_REFUND_CONTROL_ACTUALLY_PROJECTED')
    allowed = (FIRST,FAILED,m['replacement_request_id'])
    need(all(rid in allowed for rid,r in g[c.FIELD].items() if r['evaluation_candidate']=='D25'),
         'NO_FOURTH_D25_OR_OTHER_CREDIT_CONSUMER')
    row = g[c.FIELD].get(m['replacement_request_id'])
    if row is not None:
        from . import mix_core_d25_cold_source as cold
        effective=cold.effective_identity(g,m)
        selected=cold.descriptor(g,m)
        need(row['submission_nonce'] == m['replacement_submission_nonce'] and row['child_run_uuid'] == effective['run_uuid']
             and row['source_sha'] == effective['source_sha'] and row['evaluation_candidate'] == 'D25'
             and row['checkpoint_manifest_sha256'] == CHECKPOINT
             and row['planned_panel_sha256'] == g[c.FIELD][FAILED]['planned_panel_sha256']
             and row.get('D25_local_successor_sha256') == digest(m)
             and all(row.get(k)==v for k,v in selected.items())
             and (cold.DESCRIPTOR in selected or cold.DESCRIPTOR not in row), 'EXACT_NEW_ROW_MODEL_PANEL_AND_MARKER')
    from .mix_core_d25_cold_source import validated as cold_validated
    cold_validated(g,m)
    return m


def finite_control_scope(g, request_id):
    m = validated(g)
    if m is None or request_id != m['replacement_request_id']:
        return None
    return dict(schema='V5_CD_D25_FINITE_CONTROL_SCOPE_V1', marker_sha256=digest(m),
        request_id=request_id, CPU_s=7200, RPC_calls=150000, source_preparation_CPU_s=600,
        total_control_CPU_s=9000, owner_process_CPU_limit_s=1200, owner_output_limit_bytes=3*GiB,
        closing_output_reserve_bytes=512*MiB, financial_costs=deepcopy(CONTROL),
        Core_CPU_components_s=dict(cold_prepare=600, owner=1200, original_RPC=6600, closeout=600),
        wire_phase_upper_bytes=deepcopy(WIRE),
        owner_output_attribution='ORIGINAL_ALREADY_PAID_PANEL_SITE_IO',
        fresh_control_output_attribution='SEPARATE_FINITE_64_128_64_MIB_CONTROL')


def allocation(g, arm):
    m = validated(g)
    if m is None or arm != 'D25':
        return None
    from . import mix_core_accounting as c
    out = c.remaining_stage()
    for key, paid in m['paid_IO_credit'].items():
        out[key] -= paid
    return out


def paired_control(g, l):
    m = validated(g)
    need(l.get(FIELD) == m, 'SAME_CANONICAL_NEW_MARKER')
    from .mix_core_d25_cold_source import FIELD as cold_field,LOCAL05_FIELD
    need(l.get(cold_field)==g.get(cold_field),'SAME_CANONICAL_COLD_SOURCE_SUCCESSOR')
    need(l.get(LOCAL05_FIELD)==g.get(LOCAL05_FIELD),'SAME_CANONICAL_NODE_LOCAL05_SUCCESSOR')
    if m is None:
        return
    _LOCAL_paid(l,m)


def output_credit(g, row):
    m = validated(g)
    if m is None or row['request_id'] != m['replacement_request_id']:
        need('D25_local_successor_sha256' not in row, 'NO_ORPHAN_D25_CREDIT')
        return None
    need(row.get('D25_local_successor_sha256') == digest(m), 'OWN_EXACT_MARKER_BINDING')
    return m['paid_IO_credit']['Hopper_storage_bytes']


def propose(global_bytes, local_bytes, *, closed_readback, terminal_proof,
            inventory_capture, historical_model, control_plan, new_request_id,
            new_nonce, source_sha, now_epoch):
    from . import mix_core_accounting as c, mix_core_d25_continuation as prior
    g,l = json.loads(global_bytes),json.loads(local_bytes)
    c.checked(g); c.paired(g,l)
    need(FIELD not in g and FIELD not in l and not g['pending_updates'] and not g['unknown_effects']
         and all(r['status']==c.STATES[3] for r in g[c.FIELD].values()), 'ONCE_COLD_NO_UNKNOWN')
    need(0 <= now_epoch-closed_readback['observed_epoch'] <= 120
         and closed_readback['LOCAL_sha256'] == a.sha(local_bytes)
         and now_epoch < s.validated(g)['deadline_epoch']-c.GPU_UPPER, 'FRESH_CANONICAL_CLOSED_BOUNDARY')
    _closed(g,l,closed_readback,terminal_proof)
    physical = inventory_guard(inventory_capture,historical_model,now_epoch=now_epoch)
    need(control_plan['schema']=='D25_FINITE_CONTROL_SEALED_HARD_LIMIT_PLAN_V1'
         and control_plan['source_sha']==source_sha and control_plan['costs']==CONTROL
         and control_plan['wire_phase_upper_bytes']==WIRE
         and control_plan['max_source_envelope_bytes']<=16*MiB
         and control_plan['no_runtime_Core_to_Vanda_RPC'] is True
         and control_plan['no_new_model_transfer'] is True
         and re.fullmatch('[0-9a-f]{64}',control_plan['sealed_controller_sha256'])
         and re.fullmatch('[0-9a-f]{64}',control_plan['sealed_installer_sha256']),
         'SEALED_FINITE_ENFORCEMENT_NOT_ZERO_COST_ASSUMPTION')
    need(set(control_plan['wire_planned_bytes'])==set(WIRE)
         and all(type(n) is int and 0<n<=WIRE[k] for k,n in control_plan['wire_planned_bytes'].items()),
         'COMPLETE_MEASURED_OR_HARD_RESERVED_TRANSPORT_COMPONENTS')
    identities=[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce for r in identities),
         'UNUSED_NEW_IDENTITY')
    stage=c.key('D25');p=g[a.PACKAGE]
    need(p['completed_stages']==c.before_stage('D25',g)
         and l['V5_CD_Core_CPU_substages'][stage]['request_id']==FAILED
         and l['V5_CD_Core_CPU_substages'][stage]['status']=='SETTLED_KNOWN_NATIVE_TERMINAL',
         'ONLY_ORIGINAL_D25_FULL_PANEL_NEXT')
    m=dict(schema=SCHEMA,failed_request_id=FAILED,failed_source_sha=SOURCE,
        scope_grant_sha256=s.GRANT_SHA,T_CD=s.validated(g)['T_CD'],deadline_epoch=s.validated(g)['deadline_epoch'],
        failed_row_sha256={rid:digest(g[c.FIELD][rid]) for rid in (FIRST,FAILED)},
        prior_marker_sha256=digest(g[prior.FIELD]),old_prepaid_control=deepcopy(p['prepaid_control'][stage]),
        engineering_replacements_used=4,maximum_new_requests=1,adopted_epoch=now_epoch,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid=RUN,source_sha=source_sha,financial_control_paid=True,
        paid_IO_credit={k:c.remaining_stage()[k] for k in ('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes')},
        finite_control_costs=deepcopy(CONTROL),wire_phase_upper_bytes=deepcopy(WIRE),
        control_plan=deepcopy(control_plan),physical_inventory_join=physical,
        closed_readback_sha256=digest(closed_readback),terminal_proof_sha256=digest(terminal_proof),
        financial_before=dict(site_CPU_s=g['CPU_s'],site_current_bytes=g['new_storage_current_bytes'],
            site_peak_bytes=g['new_storage_peak_bytes'],local_CPU_s=deepcopy(l['CPU_s']),
            network_transfer_accounted_bytes=l['network_transfer_accounted_bytes'],
            Core_native_GPU_current_storage_bytes=l['Core_native_GPU_current_storage_bytes']),
        original_GLOBAL_sha256=a.sha(global_bytes),original_LOCAL_sha256=a.sha(local_bytes),
        all_old_costs_history_and_gross_uppers_retained=True)
    v,w=deepcopy(g),deepcopy(l)
    active=c.remaining_stage()
    for key,paid in m['paid_IO_credit'].items():active[key]-=paid
    for value in (v,w):
        value[FIELD]=deepcopy(m)
        for key,cost in active.items():value[a.PACKAGE]['remaining_resources'][key]+=cost
    for key in ('H200_s','neural_queries','optimizer_updates'):v['reserved'][key]+=active[key]
    v['CPU_reserved_s']+=active['Hopper_CPU_s'];v['CPU_s']+=CONTROL['Hopper_CPU_s']
    v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    w['GPU_reserved_s']+=active['H200_s']
    for key,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:
        w['CPU_reserved_s'][site]+=active[key];w['CPU_s'][site]+=CONTROL[key]
    w['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes']
    w['Core_native_GPU_current_storage_bytes']+=CONTROL['Core_storage_bytes']
    # Local storage is a retained upper, not a deletion/refund or model copy.
    w.setdefault('V5_CD_finite_control_local_storage_bytes',{})[new_request_id]=CONTROL['local_storage_bytes']
    v['revision']+=1;c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w))
    need(not full['cap_changes_required'],'ALL_FUTURE_MATRIX_FINAL_AND_NEW_CONTROL:'+str(full['cap_changes_required']))
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,
                original_onceIO_required=True,native_registration=False,model_or_physics_permission=False)
