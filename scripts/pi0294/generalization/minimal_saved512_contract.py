"""Exact saved-E1 inference completion and independent raw-E2 fixed512.

The old failed native job remains failed and paid. This type never grants a
new optimization to E1, relabels its512 as current effects, or resumes E2.
"""
from copy import deepcopy
import json
from . import minimal_execution as e,minimal_saved512_recovery_scope as scope

VERSION='ARX5_MINIMAL_SAVED_E1_512_FRESH_E2_512_V1'
PROFILE='vanda-arx5-minimal-saved-e1-inference-fresh-e2-fixed512-cold02'
RUN='arx5-minimal-saved-e1-inference-e2-fresh512-02'
RESULT='f900a9f5b9bf310178bb5c931f3c2812e8b77cf3b9937c6527426578212cd138'
OS_EXIT='a0e57ba904d5af68506374614734e4470ac857997d2aa2cda29f3dd2ba23da2a'
PROGRAM='1b0b22ba25925176006a84a6d08c2039baa846e3e40b63194ef86b4285a36445'
MANIFEST='053556c595b010e5064138c07266ec67bf4671298758dd762899dff1f7ca5198'
REQUEST_SHA='35a1f1ffb21768218fa16556ca3cd9de6a0f267b1c8e4fe7e6f36d478c5094b9'
SOURCE='b0509a0bf296772e8e83856a2caf259b6ae0d908'
SETTLED='d2ee85c768e3fca39d2175bff83b81d6ea8c98bd2b71a6903e6e138a63aaf9fb'
E1_CP='96b4abf08d752af9c332ac70cd4f0120bcfed0c84ea22d6650aad7869ae98a36'
E1_EXPORT='383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d'
E1_TRAIN='27d21622ec00dc1c759a959124f8af1eb49dd1a6f0d2fd9ffbd12b8173fe4ccd'
E2_CP='db93cf7720b5dcf67d2d066e160ec7ce10b27a3886a2908e8a94bfe1348861f8'
RESULT_FILE='plans/MINIMAL_SAVED512_ACTUAL_AUDIT_RESULT.private.json'
EXIT_FILE='plans/MINIMAL_SAVED512_ACTUAL_AUDIT_OS_EXIT.private.json'
PLAN='plans/MINIMAL_SAVED512_CUSTODY.private.json'
OLD_PAID=dict(optimizer_updates=516,generation_queries=110,model_loads=10)
NEW_UPPER=dict(optimizer_updates=512,generation_queries=62,model_loads=5)
KEYS=tuple(NEW_UPPER)


def binding(result_bytes,exit_bytes):
    e.need(e.sha(result_bytes)==RESULT and e.sha(exit_bytes)==OS_EXIT,
        'SAVED512_REAL_INDEPENDENT_FULL_STREAM_RESULT_AND_OS0_BYTES')
    a,z=json.loads(result_bytes),json.loads(exit_bytes)
    e.need(a['status']=='PASS_ACTUAL_SAVED_E2_QUALIFICATION_E1_FULL512_AND_ORDINARY_SERIALIZED_TENSORS_INDEPENDENT_CPU_AUDIT'
        and a['job_id']==scope.JOB and a['native_exit_status']==1
        and a['request_sha256']==REQUEST_SHA and a['source_sha']==SOURCE
        and a['original_financial_settlement_sha256']==SETTLED
        and a['ordinary_target10_table_substitution_and_all_other_tensors_exact']is True
        and a['E1_ordinary_manifest_sha256']==E1_EXPORT and a['E1_original_TRAIN_probes']==12
        and a['E1_actual_output_reconstruction_qualified']is False
        and a['E1_bad_noise_language24_preserved_not_promoted']is True
        and a['E2_formal_updates']==a['new_models_queries_updates_GPU_PBS_physics']==0
        and z['returncode']==0 and z['timeout']is False and z['foreground_wait_completed']is True,
        'SAVED512_FAILED_JOB_NOT_RELABELED_SUCCESS_OR_OUTPUT_QUALIFIED')
    for sid,step,pin in [('E1_ADAPTER_ONLY_FORMAL512',512,E1_CP),('E2_SHARED_ACTION_HEAD_QUALIFY2',2,E2_CP)]:
        c=a['saved_states'][sid]
        e.need(c['checkpoint_manifest_sha256']==pin and c['global_step']==step and c['cursor']==step*32
            and c['scheduler_horizon']==1024 and c['full_resume']is True
            and c['qualification_only']is(step==2)and c['raw_parent_files_identity']==e.training.RAW_PARENT_FILES_ID
            and c['statistics_sha256']==e.R0,'SAVED512_EXACT_INDEPENDENT_OWN_STATE_NOT_OTHER_ARM')
    w=a['streamed_witnesses'];e.need(w['paths'] and w['all_paths_rechecked']is w['all_unique_inodes_independently_rehashed']is True,
        'SAVED512_ALL_SAVED_MEMBERS_SECOND_FULL_STREAM_REQUIRED')
    return dict(schema='ARX5_SAVED512_REAL_CLOSED_INPUT_BINDING_V1',result_sha256=RESULT,OS_exit_sha256=OS_EXIT,
        program_sha256=PROGRAM,old_job_id=scope.JOB,old_request_id=scope.REQUEST,old_source_sha=SOURCE,
        old_financial_settlement_sha256=SETTLED,old_lease_sha256=scope.LEASE,E1_wrapped_manifest_sha256=E1_CP,
        E1_ordinary_manifest_sha256=E1_EXPORT,E1_TRAIN_probes_sha256=E1_TRAIN,
        E2_qualification_manifest_sha256=E2_CP,retained_old_native=deepcopy(OLD_PAID),
        E1_new_optimizer_calls=0,E2_initialization='ORIGINAL_RAW_PARENT_FRESH_OPTIMIZER',native_admission=False)


def sessions():
    rows=deepcopy(e.native_sessions()[3:]);rows[0].update(optimizer_limit=0,query_limit=25,
        sample_visits=16384,fresh_raw_parent=False,saved_own_step=512,inference_finalization_only=True)
    rows[1].update(saved_own_step=0,inference_finalization_only=False)
    return rows


def tickets():
    return [dict(index=i,session_id=s['id'],arm=s['arm'],phase=p)for i,(s,p)in enumerate(
        (s,p)for s in sessions()for p in(('MODEL_RAW','RESTORE_SAVED512','OFFICIAL_RELOAD')
        if s['inference_finalization_only']else('MODEL_RAW','FIRST_UPDATE','OFFICIAL_EXPORT')))]


def check_phase(sid,phase,counters):
    spec=next((s for s in sessions()if s['id']==sid),None)
    e.need(spec is not None and set(counters)==set(KEYS),'SAVED512_ONLY_TWO_NAMED_SESSIONS')
    expected={'MODEL_RAW':(0,0,0)}
    if spec['inference_finalization_only']:expected.update(RESTORE_SAVED512=(0,0,1),OFFICIAL_RELOAD=(0,24,2))
    else:expected.update(FIRST_UPDATE=(0,0,1),OFFICIAL_EXPORT=(512,36,1))
    e.need(phase in expected and tuple(counters[k]for k in KEYS)==expected[phase],
        'SAVED512_EXACT_PHASE_DEBITS_NO_E1_UPDATE_OR_OLD_QUERY_CREDIT')
