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
    if m is None:return None
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
    return m


def effective_identity(g,original):
    m=validated(g,original)
    return dict(source_sha=m['source_sha'],run_uuid=m['run_uuid'])if m else dict(
        source_sha=original['source_sha'],run_uuid=original['replacement_run_uuid'])


def descriptor(g,original):
    m=validated(g,original)
    return {DESCRIPTOR:digest(m)}if m else {}


def sealed_context(root,manifest,request,local,original,read,sha):
    m=validated(local,original)
    if m is None:
        need(DESCRIPTOR not in manifest and DESCRIPTOR not in request,'NO_ORPHAN_SELECTOR')
        return
    need(read(root/BINDING)==m and sha(root/BINDING)==manifest['execution']['sealed_inputs'].get(BINDING)==digest(m)
         and manifest.get(DESCRIPTOR)==request.get(DESCRIPTOR)==digest(m)
         and manifest['code']['vla_commit']==request['source_sha']==m['source_sha']
         and manifest['code']['source_seal_sha256']==m['source_seal_sha256']
         and manifest['run_uuid']==request['run_uuid']==RUN,'EXACT_IMMUTABLE_NEW_RUNTIME_BINDING')


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
