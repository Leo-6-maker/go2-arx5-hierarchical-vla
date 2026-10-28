"""Cold first512 metadata factory around the actual original submit program.

No registry/lease/ledger mutation, scheduler call or native permission occurs.
The original A40 entry stays the compute entry; its new typed context checks
the existing Vanda registry/profile rather than inventing a second launcher.
"""
from copy import deepcopy
import re
from . import minimal_execution as e

ORIGINAL_SUBMIT_SHA='be8af0be7c233e2c67e1f0659cf2f835b4653dbb1fb710f7d0e4c87ac012a67f'


def effective_queue(body,*,created_epoch,walltime_s):
    import json
    q=json.loads(body)
    e.need(q.get('source')=='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
        and q.get('requested_queue')=='auto'and q.get('destination')=='batch_gpu'
        and type(q.get('maximum_wall_s'))is int and q['maximum_wall_s']>=walltime_s
        and type(q.get('observed_epoch'))in(int,float)and 0<=created_epoch-q['observed_epoch']<=120,
        'LAUNCH_ACTUAL_EFFECTIVE_QUEUE_LIMIT_STILL_FRESH_BEFORE_REGISTER_OR_QSUB')
    return q


def template(*,source_sha,run_uuid,request_id,submission_nonce,scope_marker,python,
             raw_parent_audit_plan_sha256,frozen_input_packet_sha256,Core_measurement_capture_sha256):
    need=e.need
    need(type(source_sha)is str and re.fullmatch('[0-9a-f]{40}',source_sha)
        and e._uuid(request_id)and e._uuid(submission_nonce)and request_id!=submission_nonce
        and type(run_uuid)is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,95}',run_uuid)
        and type(python)is str and python.startswith('/'),'LAUNCH_EXPLICIT_NEW_COLD_IDENTITY_NOT_OLD_ARM')
    hashes=dict(raw_parent_audit_plan_sha256=raw_parent_audit_plan_sha256,
        frozen_input_packet_sha256=frozen_input_packet_sha256,Core_measurement_capture_sha256=Core_measurement_capture_sha256)
    need(all(type(v)is str and re.fullmatch('[0-9a-f]{64}',v)for v in hashes.values()),'LAUNCH_REAL_SEALED_INPUT_HASHES')
    from scripts.pi0294.numerics import MODE
    v=dict(e.fixed_request_fields(),source_sha=source_sha,run_uuid=run_uuid,request_id=request_id,
        submission_nonce=submission_nonce,python=python,numerical_mode=MODE,**hashes)
    for key in('T_CD','stop_new_training_epoch','deadline_epoch'):v[key]=scope_marker[key]
    # Actual manifest and native cold audit are bound only after both exist.
    # No fabricated digest here can become an actual allocation/release.
    return v


def programs(original):
    values=deepcopy(original);s=values['SUBMIT']
    e.need(e.sha(s.encode())==ORIGINAL_SUBMIT_SHA,'LAUNCH_EXACT_PREVIOUS_SUCCESSFUL_D25_SUBMIT_BYTES')
    replacements={
        'select=1:ngpus=1:ncpus=4:mem=32gb':'select=1:ngpus=1:ncpus=8:mem=64gb',
        'walltime=06:15:00':'walltime=16:30:00',
        'timeout=45':'timeout=30',
        'env=dict(PYTHONPATH=':"env=dict(PV1_RUN_MANIFEST=str(r/'RUN_MANIFEST.json'),GR00T_WBC_PROFILE="+repr(e.PROFILE)+',PYTHONPATH='}
    for before,after in replacements.items():
        e.need(s.count(before)==1,'LAUNCH_EXACT_SINGLE_ORIGINAL_SUBMIT_SEAM:'+before);s=s.replace(before,after)
    e.need(s.count('scripts.pi0294.dec1.a40_training_runtime')==1 and '#PBS -m n'in s
        and '#PBS -r n'in s and 'unknown_reply_never_retried=True'in s
        and 'PV1_REQUEST_ID='in s and 'PV1_SUBMISSION_NONCE='in s,
        'LAUNCH_ORIGINAL_ENTRY_AND_ONCE_QSUB_WRAPPER_RETAINED')
    compile(s,'<original-first512-cold-submit>','exec');values['SUBMIT']=s
    return values
