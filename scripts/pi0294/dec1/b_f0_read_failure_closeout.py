"""Accounting-only boundary: actual F0/full4096, failed post-training reader.

The successful-owner predicate stays unchanged. This permits legal effects
settlement/lease closure, never checkpoint reuse, a new run, or science GO.
"""
import math
from . import b_precuda_retry_completion as original

def need(ok,why):
    if not ok:raise PermissionError('B_F0_FAILED_READER_FINANCIAL_'+why)


def owner(packet,expected,evidence,*,now_epoch):
    original.expected(expected);original.audit(evidence,expected)
    exit_record=packet['OS_exit'];process=packet['owner_process'];launch=packet['launch'];current=packet['current_process']
    stop=packet['preserved_stop'];error=packet['failed_read_event'];rpc=packet['current_rpc']
    need(exit_record.get('schema')=='VANDA_ORIGINAL_FOREGROUND_OWNER_OS_EXIT_V1'
         and type(exit_record.get('returncode')) is int and exit_record['returncode']==1
         and exit_record.get('foreground_wait_completed') is True
         and exit_record.get('source_sha')==process.get('source_sha')==launch.get('source_sha')==original.SOURCE
         and exit_record.get('owner_process')==process and exit_record.get('wrapper_pid')==launch.get('pid')
         and launch.get('process_kind')=='ORIGINAL_FOREGROUND_EXIT_CAPTURE_WRAPPER'
         and process.get('uid')==22182 and process.get('maximum_qsub')==1
         and process.get('agent_connection_required') is False
         and process.get('pid')==original.OWNER_PID and process.get('proc_start_ticks')==original.OWNER_START
         and launch.get('pid')==original.WRAPPER_PID and launch.get('proc_start_ticks')==original.WRAPPER_START,
         'EXACT_REAL_FAILED_ORIGINAL_OWNER_AND_WRAPPER_NO_RESTART')
    for proof,pid,start in ((current,original.OWNER_PID,original.OWNER_START),
                            (packet['current_wrapper_process'],original.WRAPPER_PID,original.WRAPPER_START)):
        need(proof.get('pid')==pid and proof.get('present_same_instance') is False
             and proof.get('physical_platform')=='VANDA_A40'
             and proof.get('proc_start_ticks')!=start
             and type(proof.get('observed_epoch')) in (int,float)
             and 0<=now_epoch-proof['observed_epoch']<=120,'FRESH_BOTH_ORIGINAL_ACTORS_ABSENT')
    need(packet.get('preserved_stop_exists') is True
         and stop.get('error_type')=='RuntimeError' and stop.get('message')=='VANDA_LOCAL_CONTROL_PROGRAM_FAILED:1'
         and stop.get('automatic_mutation_retry') is False
         and error.get('event')=='ERROR' and error.get('outcome')=='READ_FAILED' and error.get('error_type')=='RuntimeError'
         and error['intent'].get('effect')=='READ_ONLY'
         and error['intent'].get('phase')=='READ_ONLY_838268923C8968CCD5E7'
         and error['intent'].get('timeout_s')==40 and error['intent'].get('automatic_write_retry') is False,
         'EXACT_FAILED_READ_NOT_UNKNOWN_MUTATION_OR_SAFETY_FAULT')
    identity=dict(run_uuid=original.RUN,job_id=original.JOB,source_sha=original.SOURCE,request_sha256=expected['request_sha256'])
    need(error['intent'].get('identity')==rpc['intent'].get('identity')==identity
         and rpc.get('status')=='CONFIRMED' and rpc['intent'].get('effect')=='HEARTBEAT'
         and packet.get('all_failed_read_events')==[error],
         'EXACT_COMPLETE_READ_FAILURE_CENSUS_AND_LAST_WRITE_CONFIRMED')
    epochs=(evidence['effects']['finished_epoch'],error['intent']['started_epoch'],error['observed_epoch'],stop['epoch'],exit_record['observed_epoch'])
    need(all(type(x) in (int,float) and math.isfinite(x) for x in epochs)
         and list(epochs)==sorted(epochs) and epochs[-1]<=now_epoch,
         'ACTUAL_COMPLETE_TRAINING_PRECEDES_FAILED_READ_AND_REAL_EXIT')
    return dict(status='F0_FULL_STATE_AUDITED_FAILED_READER_LEGAL_FINANCIAL_CLOSEOUT_ONLY',
        accounting_close_permitted=True,original_successful_owner_predicate_satisfied=False,
        failed_reader_history_must_be_retained=True,checkpoint_reuse_or_new_science_admitted=False,
        new_training_query_or_physics_admitted=False)
