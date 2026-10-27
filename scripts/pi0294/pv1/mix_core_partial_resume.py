"""One cold continuation of the ORIGINAL C50 panel after known partial effects.

The existing GLOBAL/LOCAL escrow and original onceIO own all persistence.
Eleven valid outcomes are immutable inputs; only thirteen original cards run.
Old costs/gross outputs and every predecessor row remain. No new budget.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a,successor_cd as s,partial_panel as prior
from .future_budget import _vector

FIELD='V5_CD_C50_original_eleven_valid_remaining13_continuation'
KEY='original_partial_predecessor_request_id'
BINDING='plans/V5_CD_ORIGINAL_PARTIAL_CONTINUATION.private.json'
HISTORY='plans/V5_CD_ORIGINAL_PARTIAL_HISTORY.private.json'
EVIDENCE='plans/V5_CD_ORIGINAL_PARTIAL_EVIDENCE.private.json'
INPUTS='plans/V5_CD_UNCHANGED_PAID_MODEL_INODES.private.json'
PRIOR_SHA='905b3e9c4fad8b98a5b8c2e697a68e1d899107344e0f65877ce2ac1653462ccc'
CONTROL=_vector(dict(Hopper_CPU_s=7200,local_CPU_s=7200,Hopper_storage_bytes=64*1024**2,
                    Core_storage_bytes=128*1024**2,transfer_bytes=128*1024**2))


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_ORIGINAL_PARTIAL_'+why)


def envelope(value):
    """Old requests keep their literal8h/24 semantics, even after adoption."""
    if KEY not in value:return dict(walltime_s=28800,reserved_GPU_s=28860,CPU_upper_s=80000,
        declared_process_CPU_limit_s=79760,remaining_conditions=24,query_limit=2097,
        additional_output_upper_bytes=42641095480,physical_CPU_s=93900,episodes=26,
        constructors=2108,physics_ticks=832000)
    need(value.get(KEY)==prior.FAILED and value.get('evaluation_candidate',value.get('candidate'))=='C50'
         and value.get('partial_panel_history_sha256')==prior.HISTORY_SHA
         and value.get('prior_outcomes_sha256')==PRIOR_SHA,'EXACT_C50_VALIDATED_PRIOR_IDENTITY')
    return dict(walltime_s=18000,reserved_GPU_s=18060,CPU_upper_s=40000,
        declared_process_CPU_limit_s=39760,remaining_conditions=13,query_limit=1057,
        additional_output_upper_bytes=4*1024**3,physical_CPU_s=47100,episodes=13,
        constructors=1055,physics_ticks=416000)


def physical_limits(value):
    v=envelope(value)
    return dict(CPU_s=v['physical_CPU_s'],physics_ticks=v['physics_ticks'],
                policy_episodes=v['episodes'],engine_starts=v['constructors'])


def allocation(g,record):
    marker=validate(g);need(marker is not None and record['request_id']==marker['replacement_request_id'],
                         'ONLY_REGISTERED_REMAINING13_REQUEST')
    v=prior.resources()
    for k,cost in CONTROL.items():v[k]-=cost
    return v


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_accounting as c
    r=g[c.FIELD][prior.FAILED];clock=s.validated(g)
    need(m.get('schema')=='V5_CD_EXACT_VALID11_KNOWN_PARTIAL_REMAINING13_V1'
         and m['failed_request_id']==prior.FAILED and m['failed_source_sha']==prior.SOURCE
         and m['scope_grant_sha256']==s.GRANT_SHA and m['T_CD']==clock['T_CD']
         and m['deadline_epoch']==clock['deadline_epoch'],'ONE_ORIGINAL_SCOPE_AND_CLOCK')
    need(r['status']=='NATIVE_TERMINAL'and r['native_exit_status']==1 and r['query_attempts']==574
         and r['model_load_attempts']==1 and r['unknown_query_intents']==[]
         and r['terminal_native_sha256']==c.digest(m['native_proof'])
         and r['terminal_effect_summary_sha256']==c.digest(m['effects'])
         and m['effects']['unknown_query_intents']==[],'FULL_KNOWN_NONZERO_PREDECESSOR_NOT_ZERO_START')
    h=m['history'];need(a.sha(a.raw(h))==prior.HISTORY_SHA and h['valid_conditions']==11
         and h['remaining_eligible_conditions']==13 and a.sha(a.raw(h['prior_rows']))==PRIOR_SHA,
         'EXACT_ALL_ATTEMPTS_VALID_FAILURES_AND_REMAINDER')
    need(m['resources']==prior.resources()and m['prepaid_control']==CONTROL
         and m['retained_IO']['new_native_IO_credit_adopted']is False
         and m['retained_IO']['combined_retained_upper_bytes']<=m['retained_IO']['original_Core_gross_already_paid']
         and m['all_old_costs_and_unknowns_retained']is True
         and g[a.PACKAGE]['engineering_replacements_used']==4,'EXACT_FINITE_EXISTING_RESOURCE_CLASSIFICATION')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'THREE_CLOSED_PREDECESSOR_LEASES')
    for v in m['closed_leases'].values():
        need(v['status']=='CLOSED'and v['all_actual_workers_terminal']is True
             and v['terminal_native_sha256']==r['terminal_native_sha256'],'REAL_CLOSED_PREDECESSOR')
    need(c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-13'
         and m['replacement_request_id']!=prior.FAILED
         and type(m['source_sha'])is str and re.fullmatch('[0-9a-f]{40}',m['source_sha']),
         'ONE_DISTINCT_FROZEN_COLD_CONTINUATION')
    return m


def record_binding(g,r):
    if KEY not in r:return None
    m=validate(g);need(m is not None and r['request_id']==m['replacement_request_id']
        and r['submission_nonce']==m['replacement_submission_nonce']and r['child_run_uuid']==m['replacement_run_uuid']
        and r['source_sha']==m['source_sha']and r.get('partial_continuation_sha256')==a.sha(a.raw(m)),
        'ACTUAL_ADOPTED_PARTIAL_REQUEST_NOT_CALLER_ENVELOPE')
    from .core_source_witness import row_contract
    need(row_contract(r)is not None,'NEW_PRESEALED_REMOTE_SOURCE_CUSTODY_REQUIRED')
    envelope(r);return m


def propose(global_bytes,local_bytes,*,population_bytes,evidence,closed_readback,native_proof,
            effects,retained_IO_readback,new_request_id,new_nonce,source_sha,now_epoch):
    from . import mix_core_accounting as c
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    need(FIELD not in g and FIELD not in l and g[a.PACKAGE]['engineering_replacements_used']==3
         and all(r['status']=='NATIVE_TERMINAL'for r in g[c.FIELD].values())
         and not g['unknown_effects']and not g['pending_updates'],'ONCE_COLD_KNOWN_TERMINAL_BOUNDARY')
    cap=closed_readback
    need(cap['status']=='INDEPENDENT_C12_KNOWN_PARTIAL_ONCE_SETTLED_AND_THREE_CLOSED_READBACK'
         and cap['request_id']==prior.FAILED and cap['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-cap['observed_epoch']<=120
         and now_epoch<g[s.FIELD]['deadline_epoch']-18060,'FRESH_CANONICAL_CLOSED_CAPTURE')
    h=prior.history(population_bytes,evidence,checkpoint_manifest_sha256=g[c.FIELD][prior.FAILED]['checkpoint_manifest_sha256'])
    leases={k:v['value']for k,v in cap['leases'].items()};io=prior.retained_IO(retained_IO_readback)
    need(0<=now_epoch-retained_IO_readback['observed_epoch']<=120,'FRESH_COMPLETE_RETAINED_IO_CENSUS')
    forecast=prior.forecast(global_bytes,local_bytes,validated_history=h,native_proof=native_proof,effects=effects,closed_leases=leases)
    need(forecast['shortfalls']=={},'ALL_ORIGINAL_FUTURE_MATRIX_AND_FINAL:'+str(forecast['shortfalls']))
    identities=[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce for r in identities),'NO_OLD_IDENTITY_REPLAY')
    m=dict(schema='V5_CD_EXACT_VALID11_KNOWN_PARTIAL_REMAINING13_V1',scope_grant_sha256=s.GRANT_SHA,
        failed_request_id=prior.FAILED,failed_source_sha=prior.SOURCE,native_proof=deepcopy(native_proof),effects=deepcopy(effects),
        history=h,evidence_sha256=a.sha(a.raw(dict(evidence=[{k:__import__('base64').b64encode(v).decode()for k,v in row.items()}for row in evidence]))),closed_leases=leases,retained_IO=io,resources=prior.resources(),prepaid_control=CONTROL,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-13',source_sha=source_sha,
        T_CD=g[s.FIELD]['T_CD'],deadline_epoch=g[s.FIELD]['deadline_epoch'],adopted_epoch=now_epoch,
        all_old_costs_and_unknowns_retained=True)
    v,w=deepcopy(g),deepcopy(l)
    for val in(v,w):
        val[FIELD]=deepcopy(m);p=val[a.PACKAGE];p['engineering_replacements_used']+=1
        for k,cost in m['resources'].items():p['remaining_resources'][k]+=cost-CONTROL[k]
    active={k:m['resources'][k]-CONTROL[k]for k in m['resources']}
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=active[k]
    v['CPU_reserved_s']+=active['Hopper_CPU_s'];v['CPU_s']+=CONTROL['Hopper_CPU_s']
    v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes'];v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    w['GPU_reserved_s']+=active['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:
        w['CPU_reserved_s'][site]+=active[k];w['CPU_s'][site]+=CONTROL[k]
    w['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes']
    w['Core_native_GPU_current_storage_bytes']=w.get('Core_native_GPU_current_storage_bytes',0)+CONTROL['Core_storage_bytes']
    v['revision']+=1;c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'COMPLETE_FUTURE_AFTER_PREPAY:'+str(full['cap_changes_required']))
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,original_onceIO_required=True,
                native_registration=False,model_or_physics_permission=False)


def validate_inputs(root,manifest,local):
    """Original baseline checks again on complete sealed saved evidence bytes."""
    from pathlib import Path
    import base64,os,stat
    root=Path(root)
    if KEY not in manifest:return None
    marker=validate(local);need(marker is not None,'ADOPTED_LOCAL_PARTIAL_SCOPE')
    def member(name):
        path=root/name;need(path.is_file()and not path.is_symlink()and not path.stat().st_mode&0o222,
                            'IMMUTABLE_PARTIAL_INPUT:'+name)
        b=path.read_bytes();need(a.sha(b)==manifest['execution']['sealed_inputs'].get(name),'PRESEALED_PARTIAL_INPUT:'+name)
        return b
    need(json.loads(member(BINDING))==marker,'EXACT_ONCE_ADOPTED_PARTIAL_SCOPE')
    history=json.loads(member(HISTORY));packet=json.loads(member(EVIDENCE))
    evidence=[{k:base64.b64decode(v,validate=True)for k,v in row.items()}for row in packet['evidence']]
    h=prior.history(member('plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'),evidence,
                    checkpoint_manifest_sha256=history['checkpoint_manifest_sha256'])
    need(h==history==marker['history']and a.sha(member('plans/PRIOR_EVALUATION_OUTCOMES.json'))==PRIOR_SHA,
         'UNCHANGED_ALL_ORIGINAL_ATTEMPTS_NOT_ONLY_PRIOR_ROW_ASSERTIONS')
    inputs=json.loads(member(INPUTS));input_root=Path(inputs['root'])
    need(inputs.get('checkpoint_manifest_sha256')==h['checkpoint_manifest_sha256']
         and inputs.get('original_inventory_sha256')==marker['retained_IO']['inventory_sha256']
         and input_root.is_absolute()and not input_root.is_symlink()
         and input_root.stat().st_uid==os.getuid()==2009,'EXACT_ORIGINAL_PAID_INPUT_ROOT')
    for name,row in inputs['files'].items():
        relative=Path(name);need(not relative.is_absolute()and '..'not in relative.parts,'INPUT_NAME')
        path=input_root/relative;st=path.lstat()
        need(stat.S_ISREG(st.st_mode)and not st.st_mode&0o222 and st.st_uid==2009
             and path.resolve().is_relative_to(input_root.resolve())
             and all(getattr(st,k)==row[v]for k,v in [('st_dev','dev'),('st_ino','inode'),('st_size','size'),
                 ('st_mtime_ns','mtime_ns'),('st_ctime_ns','ctime_ns')]),'EXACT_UNCHANGED_PAID_MODEL_MEMBER:'+name)
    if manifest['scope']=='V5_CD_CORE_NEURAL_EVALUATION_V1':
        view=root/'model_views/C50'
        need(view.is_symlink()and view.resolve()==input_root.resolve(),'ONE_EXPLICIT_READONLY_MODEL_REFERENCE_NO_NEW_TENSOR_LINK_OR_COPY')
    return inputs


class OutputGuard:
    """Bound only new metadata/output, preserving all paid readonly model inodes.

    At most eight actor replies between full inode censuses. Each persistent
    reply/snapshot is bounded in the same RPC code;512MiB is withheld inside
    the4GiB envelope for all three original actors and terminal metadata.
    No files, effects, unknown latches or previous results are removed.
    """
    def __init__(self,root,manifest,local):
        from pathlib import Path
        self.root=Path(root);self.inputs=validate_inputs(root,manifest,local)
        self.manifest=manifest;self.local=local;self.calls=0
    def check(self):
        import os,stat
        if self.inputs is None:return
        # Read unchanged sealed input/inode identities before every RPC; full
        # baseline evidence stays bound to the same immutable input hashes.
        from pathlib import Path
        input_root=Path(self.inputs['root'])
        for name,row in self.inputs['files'].items():
            st=(input_root/name).lstat()
            need(stat.S_ISREG(st.st_mode)and not st.st_mode&0o222
                 and (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==
                     tuple(row[k]for k in('dev','inode','size','mtime_ns','ctime_ns')),'PAID_MODEL_INPUT_CHANGED')
        self.calls+=1
        if self.calls%8!=1:return
        unique={};ignored={(r['dev'],r['inode'])for r in self.inputs['files'].values()}
        for directory,dirs,files in os.walk(self.root,followlinks=False):
            for name in files:
                path=Path(directory)/name;st=path.lstat()
                if stat.S_ISLNK(st.st_mode):continue
                need(stat.S_ISREG(st.st_mode)and st.st_uid==2009,'REGULAR_NEW_OUTPUT')
                key=(st.st_dev,st.st_ino)
                if key not in ignored:unique[key]=max(st.st_size,st.st_blocks*512)
        need(sum(unique.values())<=4*1024**3-512*1024**2,'FINITE_NEW_NEURAL_OUTPUT_WITH_TERMINAL_RESERVE')


def descriptor(marker):
    return {KEY:prior.FAILED,
        "partial_panel_history_sha256":prior.HISTORY_SHA,"prior_outcomes_sha256":PRIOR_SHA,
        "partial_continuation_sha256":a.sha(a.raw(marker))}
