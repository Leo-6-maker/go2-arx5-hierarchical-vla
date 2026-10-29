"""Deliver the proved E1 endpoint without promoting its failed E2 sibling.

This narrow closed1506193 join reuses the earlier full-state/tensor audit and
the later actual same-noise inference proof. It grants no model, training,
resubmission, Core permission or replacement of either failed native record.
"""
import base64
from copy import deepcopy
import json

from . import minimal_execution as e, minimal_saved512_contract as saved
from . import minimal_saved512_execution as execution

KEY='qualified_saved_E1_source'
ARM='E1_ADAPTER_ONLY'
SID='E1_ADAPTER_ONLY_FORMAL512'
ROW='bca2e93f38bdcca3ec4d10dc79abe418d8a5fbcc20f9b627b06bb40fcd172f28'
MARKER='31d445f0f334be1603c3f9ebbfb834381e7fb6cb81b77e4c903370009fafd3a4'
AUDIT_CANONICAL='9c72670067d9ce9270a139edf85d3805e2a92618b6aa2cad470e6efde6681585'
PAIR='aa200a7a91391c1a121a4f889648837f40217bca39fa6e4d3efe2e4c7032cd5d'
SESSION='c8ff1fb2277ec9465d1ab2eb26d7145641910f23d8acb20b81c4eb4155097253'
EQUALITY='ddd909586ce2b6d67675cf2fb197c51f5c6d6f814e5a7ef7109a4adfbcdfb4fc'
SETTLED='52e39f28f50279a47a1a5efabce32b149d0bc5da4c5e01db1e06bf77a641fe81'
LEASE='d422ff437ad7409b0042c7761816963bc5bf5eae8d0c9d6c188d90293d44fabb'
FINITE_OS0='19a161af4d73bfea2d24d921ec66da70af7f3c2b9dd43f1af1e03f0b4ee4737d'
OLD_OS1='95e41d77343401a845fc96be82a07af4ba7230b3c812d6f7837f76fa7586005b'
need=e.need


def source_control_role(control_root,manifest_bytes,*,canonical_parent):
    """Bind this source-only role, preserving the legacy parent lease bytes.

    The parent is infrastructure and predates manifest_sha256 on its lease.
    The new own ACTIVE lease must carry the actual manifest identity instead.
    No legacy C/D or other minimal-model uploader uses this narrow boundary.
    """
    import os,stat
    from pathlib import Path
    root,parent=Path(control_root),Path(canonical_parent)
    need(root.resolve()==root and parent.resolve()==parent and root.parent==parent/'control',
        'QUALIFIED_E1_OWN_SOURCE_ROLE_UNDER_CANONICAL_PARENT')
    def own(path):
        p=Path(path);s=p.lstat()
        need(p.resolve()==p and stat.S_ISREG(s.st_mode)and s.st_uid==os.getuid()
            and s.st_size<=8*1024**2,'QUALIFIED_E1_OWN_REGULAR_CONTROL_METADATA')
        return p.read_bytes()
    m=json.loads(manifest_bytes);pin=e.sha(manifest_bytes)
    need(own(root/'CONTROL_RUN_MANIFEST.json')==own(root/'RUN_MANIFEST.json')==manifest_bytes
        and m['candidate']==ARM and m['profile']=='vanda-arx5-qualified-E1-model-only-IO'
        and m['canonical_parent']==str(parent)and m['control_root']==str(root)
        and m['run_uuid']==root.name and e.sha(own(parent/'RUN_MANIFEST.json'))==m['parent_manifest_sha256'],
        'QUALIFIED_E1_MATCHED_OWN_AND_PARENT_MANIFESTS')
    pb=json.loads(own(parent/'OUTPUT_BINDING.private.json'));pl=json.loads(own(pb['lease']))
    need(pl['status']=='ACTIVE_CONTROL_PLANE_ONLY','QUALIFIED_E1_LEGACY_INFRASTRUCTURE_PARENT_ONLY')
    b=json.loads(own(root/'OUTPUT_BINDING.private.json'));lease=json.loads(own(b['lease']))
    registry=json.loads(own(b['registry']));env=json.loads(own(b['environment']))
    need(b['run_root']==str(root)and b['run_manifest_sha256']==pin
        and lease['status']=='ACTIVE'and lease['manifest_sha256']==pin and lease['run_uuid']==m['run_uuid']
        and lease['role']==b['role']and lease['CPU_upper_s']==900,
        'QUALIFIED_E1_REAL_OWN_ACTIVE900CPU_LEASE_NO_PARENT_SCHEMA_EDIT')
    rows=[v for v in registry['entries']if v.get('role')==b['role']]
    profiles=[v for v in env['profiles']if v.get('name')==m['profile']]
    need(len(rows)==len(profiles)==1 and rows[0]['path']==str(root)and rows[0]['run_uuid']==m['run_uuid']
        and profiles[0]['role']==b['role']and profiles[0]['manifest']==str(root/'RUN_MANIFEST.json')
        and profiles[0]['model_or_optimizer_on_login']is False,
        'QUALIFIED_E1_STANDARD_SOURCE_ROLE_PROFILE_AND_NO_NEURAL_PERMISSION')
    return dict(own_manifest_sha256=pin,parent_manifest_sha256=m['parent_manifest_sha256'],
        parent_lease_unchanged=True,new_model_permission=False)


def source(g,evidence):
    # This join also runs inside remaining-resource projections. Do not call
    # the whole ledger validator here: it projects these same delivery costs.
    # Native entry points below still validate the complete ledger first.
    index,row=execution.basic(g)
    from .minimal_saved512_terminal import validate_terminal
    validate_terminal(g,row,index)
    need(e.sha(e.raw(row))==ROW and e.sha(e.raw(index))==MARKER
        and row['status']==e.STATES[2]and row['native_exit_status']==1
        and row['scientific_endpoint_complete']is False
        and index['status']=='INCOMPLETE'and index['completed_sessions']==[SID]
        and g['pending_updates']==g['unknown_effects']==[],
        'QUALIFIED_E1_EXACT_KNOWN_PARTIAL122_NOT_WHOLE_JOB_SUCCESS')
    t=index['terminal'];session=t['effects']['sessions'][SID]
    need(session['status']=='COMPLETE'and session['optimizer_updates']==0
        and session['generation_queries']==25 and session['model_loads']==3
        and session['terminal_receipt_sha256']==SESSION,
        'QUALIFIED_E1_REAL_COMPLETED_INFERENCE_COUNTS_NOT_E2')
    need(type(evidence)is dict and set(evidence)=={'audit','pair_base64','native_readback'},
        'QUALIFIED_E1_EXPLICIT_FULL_STATE_OUTPUT_AND_NATIVE_READBACK')
    audit=evidence['audit']
    canonical=(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    need(e.sha(canonical)==AUDIT_CANONICAL
        and audit['E1_ordinary_manifest_sha256']==saved.E1_EXPORT
        and audit['ordinary_target10_table_substitution_and_all_other_tensors_exact']is True,
        'QUALIFIED_E1_ORIGINAL_FULL_STREAM_AUDIT_CONTENT_NOT_NEW_SELF_ATTESTATION')
    pair_raw=base64.b64decode(evidence['pair_base64'],validate=True)
    need(e.sha(pair_raw)==PAIR,'QUALIFIED_E1_ORIGINAL_INDEPENDENT_ACTUAL_OUTPUT_PAIR')
    pair=json.loads(pair_raw)
    need(pair['E1_session_terminal_sha256']==SESSION
        and pair['native_comparison_sha256']==EQUALITY
        and pair['saved_state_audit_sha256']==saved.RESULT
        and pair['saved_full_checkpoint_manifest_sha256']==saved.E1_CP
        and pair['ordinary_export_manifest_sha256']==saved.E1_EXPORT
        and pair['same_actual_noise']is pair['same_observation']is True
        and pair['maximum_differences']==dict(physical_max_abs=0.,normalized_max_abs=0.)
        and pair['source_sha']==row['source_sha']and pair['request_sha256']==row['request_sha256'],
        'QUALIFIED_E1_ACTUAL_IDENTICAL_NOISE_INPUT_AND_FULL_OUTPUTS')
    current=evidence['native_readback'];files=current['files']
    pins=dict(full_state_audit=saved.RESULT,full_state_audit_OS0=saved.OS_EXIT,
        E1_session=SESSION,E1_equality=EQUALITY,settled=SETTLED,
        science_lease=LEASE,finite_OS0=FINITE_OS0,old_owner_OS1=OLD_OS1)
    need(all(files[k]['sha256']==v for k,v in pins.items())
        and current['old_owner_and_wrapper_terminal']is True
        and current['all_original_witness_stats_unchanged']is True
        and current['current_witness_count']==len(audit['streamed_witnesses']['paths'])
        and current['new_models_queries_updates_GPU_PBS']==0
        and pair['observed_epoch']<=row['settled_epoch']<=current['observed_epoch'],
        'QUALIFIED_E1_ACTUAL_CLOSED_FILES_AND_UNCHANGED_FULL_STREAM_WITNESSES')
    for key,pin in(('science_lease',LEASE),('settled',SETTLED),('finite_OS0',FINITE_OS0)):
        body=base64.b64decode(files[key]['base64'],validate=True)
        need(e.sha(body)==pin,'QUALIFIED_E1_CLOSED_NATIVE_BYTES:'+key)
    lease=json.loads(base64.b64decode(files['science_lease']['base64']))
    settled=json.loads(base64.b64decode(files['settled']['base64']))
    final=json.loads(base64.b64decode(files['finite_OS0']['base64']))
    need(lease['status']=='CLOSED'and lease['run_uuid']==row['run_uuid']
        and lease['manifest_sha256']==row['run_manifest_sha256']
        and lease['terminal_receipt_sha256']==SETTLED
        and all(settled[k]==t[k]for k in('native','effects','audit'))
        and final['returncode']==0 and final['timeout']is False
        and final['finished_epoch']>=lease['closed_epoch']>=row['settled_epoch'],
        'QUALIFIED_E1_FINITE_TERMINAL_OS0_NOT_RELABELING_DEAD_OWNER_OS1')
    cp=deepcopy(audit['saved_states'][SID]);cp.update(TRAIN_probes=12,language_probes=24,
        official_inference_manifest_sha256=saved.E1_EXPORT,
        original_training_job_id=saved.scope.JOB,current_job_optimizer_updates=0)
    return index,row,cp,audit


def closed_endpoint(global_bytes,lease_bytes,settled_bytes,observation,*,arm):
    need(arm==ARM,'QUALIFIED_E1_DOES_NOT_ADMIT_PARTIAL_E2')
    g=json.loads(global_bytes)
    from scripts.pi0294.pv1.neural_ledger import check
    check(g)
    index,row,cp,audit=source(g,observation[KEY])
    need(observation['GLOBAL_sha256']==e.sha(global_bytes)
        and e.sha(lease_bytes)==LEASE and e.sha(settled_bytes)==SETTLED,
        'QUALIFIED_E1_SAME_CLOSED_GLOBAL_AND_ACTUAL_LEASE_SETTLEMENT')
    return dict(GLOBAL_sha256=e.sha(global_bytes),request_id=row['request_id'],
        job_id=row['job_id'],source_sha=saved.SOURCE,
        inference_qualification_source_sha=row['source_sha'],
        run_manifest_sha256=row['run_manifest_sha256'],arm=arm,endpoint=cp,audit=audit,
        lease_sha256=LEASE,settled_sha256=SETTLED,
        allocation_scientific_endpoint_complete=False,only_proved_E1_admitted=True)
