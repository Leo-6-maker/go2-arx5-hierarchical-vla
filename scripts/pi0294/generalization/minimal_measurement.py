"""Existing Core measurement evidence for the first512 formal-training gate.

Consumes original bytes and the mature population/audit/causal parser. A valid
policy failure qualifies the measurement chain; success is never a gate. This
is not another neural/physical run, a DEV-data training input, or a new score.
"""
import base64,json
from . import minimal_adoption as a
from scripts.pi0294.pv1 import baseline,mix_core_accounting as core,mix_accounting as original


def validate(g,capture_bytes):
    a.need(type(capture_bytes)is bytes,'MEASUREMENT_REAL_BYTES_REQUIRED')
    cap=json.loads(capture_bytes)
    a.need(cap.get('schema')=='ARX5_MINIMAL_EXISTING_CORE_MEASUREMENT_CAPTURE_V1'
        and cap.get('candidate')=='C50','MEASUREMENT_EXISTING_C50_ROLE_NOT_TRAIN_DATA')
    names={'population','result','adjudication','audit','model_ready','causal_chain',
           'core_causal','hopper_causal'}
    a.need(set(cap.get('files',{}))==names,'MEASUREMENT_COMPLETE_ORIGINAL_AUDIT_INPUTS')
    files={}
    for k,r in cap['files'].items():
        b=base64.b64decode(r['base64'],validate=True)
        a.need(len(b)==r['bytes']and a.sha(b)==r['sha256'],'MEASUREMENT_EXACT_BYTES:'+k);files[k]=b
    ready=json.loads(files['model_ready']);identity=ready['identity'];q=ready['native_pre_model_qualification']
    from scripts.pi0294.pv1.causal_audit import join
    a.need(join(json.loads(files['core_causal']),json.loads(files['hopper_causal']))
        ==json.loads(files['causal_chain']),'MEASUREMENT_RECOMPUTE_BOTH_ORIGINAL_CAUSAL_PROOFS')
    row=g[core.FIELD].get(cap.get('request_id'),{})
    a.need(row.get('evaluation_candidate')=='C50'and row.get('native_handle')==ready['job_id']
        and row.get('child_run_uuid')==ready['run_uuid']
        and row.get('checkpoint_manifest_sha256')==identity['checkpoint_manifest_sha256']
        and row.get('source_sha')==identity['relay_identity']['source_sha']
        and identity['global_step']==1024 and identity['statistics_sha256']==original.STATS
        and ready['status']=='PV1_NATIVE_MODEL_READY'
        and q['status']=='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
        and q['heartbeat_elapsed_s']>=600 and q['transport_cycles']==80,
        'MEASUREMENT_SAME_REGISTERED_ACTUAL_MODEL_AND_ORIGINAL_NATIVE600')
    evidence={k:files[k]for k in ('result','adjudication','audit','model_ready','causal_chain')}
    history=baseline.partial_history(files['population'],candidate='C50',evidence=[evidence],
                                    checkpoint_manifest_sha256=identity['checkpoint_manifest_sha256'])
    a.need(history['valid_conditions']==1,'MEASUREMENT_ONE_REAL_VALID_CONDITION_NOT_ENGINEERING')
    return dict(status='PASS_EXISTING_CORE_MEASUREMENT_RAW_AND_TWO_SITE_CAUSAL_CHAIN',
        request_id=cap['request_id'],checkpoint_manifest_sha256=identity['checkpoint_manifest_sha256'],
        capture_sha256=a.sha(capture_bytes),native_qualification_receipt_sha256=q['receipt_sha256'],
        valid_conditions=1,success_rate_is_not_a_gate=True,
        condition_result_sha256=history['condition_result_sha256'],
        independent_audit_sha256=history['independent_audit_sha256'],
        new_model_queries_updates_or_physics=0,usable_as_training_input=False)
