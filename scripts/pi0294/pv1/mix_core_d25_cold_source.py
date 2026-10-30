"""One honest source successor for an unstarted, already paid D25 panel.

The original financial marker/request/nonce and all failed rows are immutable.
This module neither reserves resources nor installs/starts anything. Native
onceIO, full source readback, closed unused roles and fresh admission still
belong to the existing owner. A new source is never labelled as old66 bytes.
"""
import json,re
from copy import deepcopy
from . import mix_accounting as a
from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA

FIELD='V5_CD_D25_unstarted_cold_source_successor'
SCHEMA='V5_CD_D25_UNSTARTED_COLD_SOURCE_SUCCESSOR_V1'
BINDING='plans/V5_CD_D25_COLD_SOURCE_SUCCESSOR.private.json'
DESCRIPTOR='D25_cold_source_successor_sha256'
LOCAL05_FIELD=FIELD+'_node_local05'
LOCAL05_SCHEMA='V5_CD_D25_CLOSED04_NODE_LOCAL05_PRESTART_REBIND_V1'
LOCAL05_RUN='v5-cd-d25-n294101-05'
MAINTENANCE_AUTHORITY='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b'
OLD_SOURCE='66c370aef2b53627fc417513b721059e04b7e2ca'
OLD_RUN='v5-cd-d25-n294101-03'
RUN='v5-cd-d25-n294101-04'
REQUEST='3e774da8-1f75-4930-b732-3f0409f3d5a4'
GLOBAL='f2c2b6da49fb8a432947cc7b8909cad3fa4385cfea36d5b1888cd657b6c723de'
LOCAL='3d66e6d1abd23083c8d4a26c8092f6e4ca0dd35af4a3133b186d9a83f2ce5051'


def need(ok,why):
    if not ok:raise PermissionError('D25_UNSTARTED_COLD_SOURCE_'+why)


def digest(v):return a.sha(a.raw(v))


def validated(g,original):
    m=g.get(FIELD)
    if m is None:
        need(LOCAL05_FIELD not in g,'NODE_LOCAL05_NO_ORPHAN_SUCCESSOR')
        return None
    need(original is not None and original['source_sha']==OLD_SOURCE
         and original['replacement_request_id']==REQUEST
         and original['replacement_run_uuid']==OLD_RUN
         and m['schema']==SCHEMA and m['original_financial_marker_sha256']==digest(original)
         and m['request_id']==REQUEST and m['submission_nonce']==original['replacement_submission_nonce']
         and m['old_source_sha']==OLD_SOURCE and m['old_run_uuid']==OLD_RUN and m['run_uuid']==RUN,
         'EXACT_ORIGINAL_UNUSED_IDENTITY_AND_FINANCIAL_BYTES')
    need(re.fullmatch('[0-9a-f]{40}',m['source_sha']) and m['source_sha']!=OLD_SOURCE
         and re.fullmatch('[0-9a-f]{64}',m['source_seal_sha256'])
         and m['original_numerical_table_sha256']==ORIGINAL_NUMERICAL_TABLE_SHA
         and m['before_GLOBAL_sha256']==GLOBAL and m['before_LOCAL_sha256']==LOCAL
         and m['new_request_count']==m['new_budget']==m['new_tensor_transfers']==0
         and m['native_admission']is False and m['old_cold_roles_retained']is True
         and original['T_CD']<=m['adopted_epoch']<original['deadline_epoch']-28860,
         'HONEST_NEW_SOURCE_SAME_MODEL_PANEL_CLOCK_AND_COSTS')
    proof=m['closed_cold_roles'];need(set(proof)=={'owner','physical','neural'},'ALL_THREE_UNUSED_ROLES')
    for role,item in proof.items():
        need(item['status']=='ACTUAL_UNUSED_COLD_ROLE_CLOSED_WITH_NO_ACTOR'
             and item['source_sha']==OLD_SOURCE and item['request_id']==REQUEST
             and item['root'].startswith('/')and item['lease'].startswith('/')
             and all(re.fullmatch('[0-9a-f]{64}',item[k])for k in
                 ('manifest_sha256','closed_lease_sha256','native_intent_sha256','native_receipt_sha256','inventory_sha256'))
             and item['native_start_records']==item['live_actors']==item['model_calls']==item['physics_ticks']==0,
             'EXACT_UNUSED_ROLE_TERMINAL_NOT_A_FAILED_SCIENCE_RETRY:'+role)
    return _node_local05(g,original,m)if LOCAL05_FIELD in g else m


def _node_local05(g,original,prior):
    """One authorized successor of closed04; no new scientific allocation."""
    from . import mix_core_accounting as c
    m=g[LOCAL05_FIELD];old=m['preserved_prestart_row']
    need(m['schema']==LOCAL05_SCHEMA and m['maintenance_authority_sha256']==MAINTENANCE_AUTHORITY
         and m['original_financial_marker_sha256']==digest(original)
         and m['prior_cold_marker_sha256']==digest(prior)
         and m['request_id']==REQUEST and m['submission_nonce']==original['replacement_submission_nonce']
         and m['old_run_uuid']==RUN and m['old_source_sha']==prior['source_sha']
         and m['run_uuid']==LOCAL05_RUN and m['source_sha']!=prior['source_sha']
         and re.fullmatch('[0-9a-f]{40}',m['source_sha'])
         and re.fullmatch('[0-9a-f]{64}',m['source_seal_sha256'])
         and m['original_numerical_table_sha256']==ORIGINAL_NUMERICAL_TABLE_SHA
         and m['closed_cold_roles']==prior['closed_cold_roles'],
         'NODE_LOCAL05_ORIGINAL_FINANCIAL_SOURCE_AND_EXACT_AUTHORIZED_SUCCESSOR')
    c.validate_row(old)
    need(digest(old)==m['preserved_prestart_row_sha256']and old['request_id']==REQUEST
         and old['source_sha']==prior['source_sha']and old['child_run_uuid']==RUN
         and old['submission_nonce']==m['submission_nonce']and old['status']==c.STATES[0]
         and old['native_handle']is None and not old.get('charged_effects')
         and not old.get('query_attempts',0)and not old.get('model_load_attempts',0)
         and not old.get('native_process')and REQUEST not in g.get('V5_CD_Core_finite_delegations',{}),
         'NODE_LOCAL05_ONLY_PROVED_UNSTARTED_ROW_NOT_A_NATIVE_RETRY_OR_REBIND')
    need(set(m['closed04_roles'])=={'owner','physical','neural'},'NODE_LOCAL05_ALL_THREE_CLOSED04_ROLES')
    for role,item in m['closed04_roles'].items():
        record=item['record']
        need(record['status']=='ACTUAL_UNUSED_COLD_ROLE_CLOSED_WITH_NO_NATIVE_ACTOR'
             and record['role']==role and record['source_sha']==prior['source_sha']
             and record['request_id']==REQUEST and record['root']==item['root']
             and record['root'].endswith('/'+('v5-cd-d25-eval-owner-04'if role=='owner'else RUN))
             and record['manifest_sha256']==item['manifest_sha256']
             and record['closed_lease_sha256']==item['closed_lease_sha256']
             and digest(record)==item['record_sha256']and record['live_actors']==[]
             and record['models_queries_physics']==0 and not any(record['native_effect_traces'].values())
             and record['old_ledger_rows_or_costs_rewritten']is False
             and record['old_NATIVE_TERMINAL_row_fabricated']is False
             and record['physical_metadata_CPU_upper_s_carried']==300
             and record['physical_unused_CPU_reserved_s_retained']==93600,
             'NODE_LOCAL05_EXACT_CLOSED04_TERMINAL_BYTES:'+role)
    need(m['closed04_roles']['neural']['manifest_sha256']==old['run_manifest_sha256']
         and m['new_request_count']==m['new_native_budget']==m['new_tensor_transfers']==0
         and m['old_cold_roles_retained']is True and m['native_admission']is False
         and prior['adopted_epoch']<=m['adopted_epoch']<min(original['deadline_epoch']-28860,1793404800),
         'NODE_LOCAL05_RETAIN_CLOSED_ROLES_COSTS_SCOPE_AND_BOTH_CLOCKS')
    return m


def runtime_marker(g,original,*,request_id=None):
    from .mix_core_d25_admission_repair import validate as repaired
    continuation=repaired(g,original)
    if continuation and request_id in(None,continuation['replacement_request_id']):return continuation
    return validated(g,original)


def effective_identity(g,original,*,request_id=None):
    m=runtime_marker(g,original,request_id=request_id)
    return dict(source_sha=m['source_sha'],run_uuid=m['run_uuid'])if m else dict(
        source_sha=original['source_sha'],run_uuid=original['replacement_run_uuid'])


def descriptor(g,original,*,request_id=None):
    m=runtime_marker(g,original,request_id=request_id)
    return {DESCRIPTOR:digest(m)}if m else {}


def sealed_context(root,manifest,request,local,original,read,sha):
    m=runtime_marker(local,original,request_id=request['request_id'])
    if m is None:
        need(DESCRIPTOR not in manifest and DESCRIPTOR not in request,'NO_ORPHAN_SELECTOR')
        return
    need(read(root/BINDING)==m and sha(root/BINDING)==manifest['execution']['sealed_inputs'].get(BINDING)==digest(m)
         and manifest.get(DESCRIPTOR)==request.get(DESCRIPTOR)==digest(m)
         and manifest['code']['vla_commit']==request['source_sha']==m['source_sha']
         and manifest['code']['source_seal_sha256']==m['source_seal_sha256']
         and manifest['run_uuid']==request['run_uuid']==m['run_uuid'],'EXACT_IMMUTABLE_NEW_RUNTIME_BINDING')


def prepare_node_local05(global_bytes,local_bytes,*,authority_bytes,closed_readback,
                         source_seal_bytes,frozen,source_readback,now_epoch):
    """Pure assembly view; never persist this view without finalizing peers."""
    from . import mix_core_accounting as c,mix_core_d25_local_successor as paid
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    original=paid.validated(g);prior=validated(g,original);old=g[c.FIELD][REQUEST]
    need(LOCAL05_FIELD not in g and LOCAL05_FIELD not in l
         and prior==g[FIELD]and prior['run_uuid']==RUN
         and a.sha(authority_bytes)==MAINTENANCE_AUTHORITY
         and json.loads(authority_bytes)['direct_answer']=='正式批准第5节全文并执行'
         and old==l[c.FIELD][REQUEST]
         and g['pending_updates']==g['unknown_effects']==l['pending_queries']==l['pending_updates']==l['unknown_effects']==[],
         'NODE_LOCAL05_ONCE_APPROVED_COLD_PAIRED_BOUNDARY')
    need(closed_readback['status']=='INDEPENDENT_CLOSED04_BYTES_LOCAL_AND_REGISTRY_HASHES_QUALIFIED_NEW05_ABSENT'
         and closed_readback['LOCAL_sha256']==a.sha(local_bytes)
         and closed_readback['actors']==[]and closed_readback['new05_roots_absent']is True
         and closed_readback['new_mutations']==0
         and 0<=now_epoch-closed_readback['observed_epoch']<=120,
         'NODE_LOCAL05_FRESH_CLOSED04_AND_ABSENT_NEW_ROLES')
    seal=json.loads(source_seal_bytes)
    need(len(frozen)==15 and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(name)==pin for name,pin in frozen.items()),
         'NODE_LOCAL05_ALL_FIFTEEN_ORIGINAL_NUMERICAL_FILE_BYTES')
    need(source_readback['status']=='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK'
         and source_readback['source_sha']==seal['source_sha']
         and source_readback['source_seal_sha256']==a.sha(source_seal_bytes)
         and source_readback['file_sha256']==seal['files']
         and source_readback['hostname']=='corelab-4080s'and source_readback['uid']==2009
         and source_readback['model_calls']==source_readback['physics_ticks']==0
         and 0<=now_epoch-source_readback['observed_epoch']<=120,
         'NODE_LOCAL05_ACTUAL_FULL_NODE_LOCAL_SOURCE_BYTE_READBACK')
    numerical_identity=l[FIELD]['original_numerical_table_sha256']
    need(numerical_identity==ORIGINAL_NUMERICAL_TABLE_SHA,'NODE_LOCAL05_ORIGINAL_NUMERICAL_IDENTITY')
    m=dict(schema=LOCAL05_SCHEMA,maintenance_authority_sha256=a.sha(authority_bytes),
        original_financial_marker_sha256=digest(original),prior_cold_marker_sha256=digest(prior),
        request_id=REQUEST,submission_nonce=old['submission_nonce'],old_run_uuid=RUN,
        old_source_sha=prior['source_sha'],run_uuid=LOCAL05_RUN,source_sha=seal['source_sha'],
        source_seal_sha256=a.sha(source_seal_bytes),original_numerical_table_sha256=numerical_identity,
        before_GLOBAL_sha256=a.sha(global_bytes),before_LOCAL_sha256=a.sha(local_bytes),
        preserved_prestart_row=deepcopy(old),preserved_prestart_row_sha256=digest(old),
        preserved_physical_CPU_stage=deepcopy(l['V5_CD_Core_CPU_substages'][c.key('D25')]),
        closed04_roles=deepcopy(closed_readback['roles']),closed04_readback_sha256=digest(closed_readback),
        closed_cold_roles=deepcopy(prior['closed_cold_roles']),
        full_source_readback_sha256=digest(source_readback),adopted_epoch=now_epoch,
        new_request_count=0,new_native_budget=0,new_tensor_transfers=0,
        old_cold_roles_retained=True,native_admission=False)
    v,w=deepcopy(g),deepcopy(l)
    for value in(v,w):
        value[LOCAL05_FIELD]=deepcopy(m)
        # Private assembler input only. The final proposal restores this same
        # request with its real new immutable manifest; no intermediate write.
        del value[c.FIELD][REQUEST]
    validated(v,original);c.checked(v);c.paired(v,w)
    return dict(assembly_global=v,assembly_LOCAL=w,before_global=deepcopy(g),before_LOCAL=deepcopy(l),
        intermediate_view_may_be_persisted=False,native_admission=False)


def finalize_node_local05(plan,assembly):
    """Rebind the one unused intent, preserving original bytes and all costs."""
    from . import mix_core_accounting as c
    v,w=deepcopy(plan['assembly_global']),deepcopy(plan['assembly_LOCAL'])
    m=v[LOCAL05_FIELD];old=m['preserved_prestart_row'];row=assembly['record']
    mutable={'source_sha','run_manifest_sha256','request_sha256','cold_input_audit_sha256',
             'created_epoch','child_run_uuid',DESCRIPTOR}
    changed_protected=sorted(k for k in set(row)|set(old)
        if k not in mutable and (k not in row or k not in old or row[k]!=old[k]))
    need(set(row)==set(old)and all(row[k]==old[k]for k in old if k not in mutable)
         and row['status']==c.STATES[0]and row['native_handle']is None
         and row['request_id']==REQUEST and row['child_run_uuid']==LOCAL05_RUN
         and row['source_sha']==m['source_sha']and row[DESCRIPTOR]==digest(m)
         and assembly['rows']['hopper']['manifest_sha256']==row['run_manifest_sha256']
         and assembly['request_sha256']==row['request_sha256'],
         'NODE_LOCAL05_ONLY_NEW_SOURCE_AND_IMMUTABLE_COLD_IDENTITY_FIELDS:'+repr(changed_protected))
    stage=deepcopy(m['preserved_physical_CPU_stage'])
    stage.update(run_uuid=LOCAL05_RUN,source_sha=m['source_sha'],
        run_manifest_sha256=assembly['rows']['core']['manifest_sha256'])
    for value in(v,w):value[c.FIELD][REQUEST]=deepcopy(row)
    w['V5_CD_Core_CPU_substages'][c.key('D25')]=stage;v['revision']+=1
    before_g,before_l=plan['before_global'],plan['before_LOCAL']
    need(all(v[k]==x for k,x in before_g.items()if k not in(c.FIELD,'revision'))
         and all(w[k]==x for k,x in before_l.items()if k not in(c.FIELD,'V5_CD_Core_CPU_substages')),
         'NODE_LOCAL05_NO_COST_CAP_HOLD_HISTORY_OR_CLOCK_REWRITE')
    c.checked(v);c.paired(v,w)
    return dict(proposed_global=v,proposed_LOCAL=w,original_onceIO_required=True,
        new_native_requests=0,native_admission=False,closed04_roles_reopened=False)


def propose(global_bytes,local_bytes,*,source_seal_bytes,frozen,source_readback,
            closed_cold_roles,fresh_readback,now_epoch):
    from . import mix_core_accounting as c,mix_core_d25_local_successor as old
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    original=old.validated(g)
    need(FIELD not in g and FIELD not in l and a.sha(global_bytes)==GLOBAL and a.sha(local_bytes)==LOCAL
         and original is not None and original['replacement_request_id']==REQUEST
         and REQUEST not in g[c.FIELD]and REQUEST not in l[c.FIELD]
         and g['unknown_effects']==g['pending_updates']==l['unknown_effects']==l['pending_updates']==l['pending_queries']==[]
         and all(r['status']=='NATIVE_TERMINAL'for r in g[c.FIELD].values())
         and all(r['status']in a.TERMINAL for r in g['PBS_submissions']),
         'ONCE_ALL_NATIVE_TERMINAL_BOUNDARY_NO_REGISTERED_RESTART')
    need(0<=now_epoch-fresh_readback['observed_epoch']<=120
         and fresh_readback['GLOBAL_sha256']==GLOBAL and fresh_readback['LOCAL_sha256']==LOCAL
         and fresh_readback['new_mutations']==0,'FRESH_EXACT_CANONICAL_PAIR')
    seal=json.loads(source_seal_bytes)
    need(len(frozen)==15 and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and all(seal['files'].get(n)==v for n,v in frozen.items()),'ORIGINAL_FIFTEEN_OVERLAY_BYTES')
    need(source_readback['status']=='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK'
         and source_readback['source_sha']==seal['source_sha']
         and source_readback['source_seal_sha256']==a.sha(source_seal_bytes)
         and source_readback['file_sha256']==seal['files']
         and source_readback['hostname']=='corelab-4080s'and source_readback['uid']==2009
         and 0<=now_epoch-source_readback['observed_epoch']<=120
         and source_readback['model_calls']==source_readback['physics_ticks']==0,
         'ACTUAL_ALL_SOURCE_BYTES_NOT_GIT_HEAD_OR_PREPARED_PASS')
    v,w=deepcopy(g),deepcopy(l)
    m=dict(schema=SCHEMA,request_id=REQUEST,submission_nonce=original['replacement_submission_nonce'],
        original_financial_marker_sha256=digest(original),old_source_sha=OLD_SOURCE,old_run_uuid=OLD_RUN,
        source_sha=seal['source_sha'],source_seal_sha256=a.sha(source_seal_bytes),run_uuid=RUN,
        original_numerical_table_sha256=ORIGINAL_NUMERICAL_TABLE_SHA,
        before_GLOBAL_sha256=GLOBAL,before_LOCAL_sha256=LOCAL,closed_cold_roles=deepcopy(closed_cold_roles),
        full_source_readback_sha256=digest(source_readback),adopted_epoch=now_epoch,
        new_request_count=0,new_budget=0,new_tensor_transfers=0,native_admission=False,old_cold_roles_retained=True)
    for value in(v,w):value[FIELD]=deepcopy(m)
    v['revision']+=1;validated(v,original);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    before=first512_forecast(global_bytes,local_bytes);after=first512_forecast(a.raw(v),a.raw(w))
    need(before['dimensions']==after['dimensions']and after['cap_changes_required']=={},'ALL14_UNCHANGED_MATRIX_FINAL_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=after,
        original_onceIO_required=True,native_registration=False,native_admission=False)
