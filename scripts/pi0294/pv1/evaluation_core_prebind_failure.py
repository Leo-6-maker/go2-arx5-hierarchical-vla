"""Settle one observed systemd failure before native process binding.

No process handle, proc-start tick, qualification, retry or admission is invented.
The exact failed unit has an actual exit and a complete zero-model journal.
"""
from copy import deepcopy
import math,re
from . import evaluation_core_B24_1000 as late
RID=late.RID
REQUEST_SHA='75a025f22bd27f35436eb2695eb197df603fad71da6053d6685d089dec342a30'
SOURCE='45ed25f978f27fcfab5cc70189037d3230e155e1'
FINAL='eaab9ce04d124bc6f1fb25753482d1871627733cffe6693e3f74212f826194f0'
KIND='SYSTEMD_UNIT_FAILED_BEFORE_NATIVE_PROCESS_BINDING'
def need(ok,why):
 if not ok:raise PermissionError('PV1_CORE_PREBIND_FAILURE_'+why)
def validate_row(r):
 need(r.get('request_id')==RID and r.get('request_sha256')==REQUEST_SHA and r.get('source_sha')==SOURCE
      and r.get('late_start_grant_sha256')==late.GRANT_SHA and r.get('native_handle') is None
      and r.get('native_start_rejected_before_binding') is True and r.get('terminal_identity_kind')==KIND
      and r.get('native_exit_status')==1 and r.get('query_attempts')==r.get('model_load_attempts')==0
      and r.get('unknown_query_intents')==[] and not any(k in r for k in ('pid','proc_start','boot_id','process_started_epoch')),
      'EXACT_FAILED_UNBOUND_IDENTITY_NOT_NATIVE_QUALIFICATION')
 need(r.get('actual_failed_systemd_InvocationID')=='9908918021a640d4896fdc46c93d590c'
      and r.get('actual_failed_systemd_ExecMainPID')==258302,'EXACT_TERMINATED_UNIT_INVOCATION')
 return r
def propose(g,l,*,proof,now_epoch):
 from . import evaluation_core_accounting as c
 c.checked(g);c.scope.validate_pair(g,l);r=g[c.FIELD][RID];p=proof
 need(r['status']==c.STATES[0] and r['native_handle'] is None and c.progress(r)==(0,0,0),'ONCE_ORIGINAL_UNBOUND_ESCROW')
 need(p['schema']=='PV1_CORE_PREBIND_SYSTEMD_KNOWN_ZERO_TERMINAL_V1' and p['request_id']==RID and p['request_sha256']==REQUEST_SHA
      and p['source_sha']==SOURCE and p['actual_NATIVE_PROCESS_FINAL_sha256']==FINAL
      and p['native_unit']=='pv1-v4a-b4096-n294101-read-recovery-02-neural.service'
      and p['native_unit_MainPID']==0 and p['native_unit_ActiveState']=='failed' and p['native_exit_status']==1
      and p['InvocationID']=='9908918021a640d4896fdc46c93d590c' and p['ExecMainPID']==258302
      and p['native_process_started_receipt_exists'] is False and p['native_model_or_phase_release_exists'] is False
      and p['all_actual_workers_terminal'] is True and p['stable_inventory'] is True
      and p['unknown_effects']==[] and p['actual_model_loads']==p['actual_queries']==p['actual_physics']==p['actual_training_updates']==0,
      'ACTUAL_EXACT_EXIT_AND_FULL_KNOWN_ZERO_NOT_TIMEOUT_OR_PID_ABSENCE')
 need(r['created_epoch']<=p['started_epoch']<=p['finished_epoch']<=now_epoch
      and 0<=now_epoch-p['observed_epoch']<=120,'ACTUAL_UNIT_WALL_AND_FRESH_READBACK')
 gpu=p['conservative_GPU_s'];cpu=p['conservative_CPU_s'];output=p['new_resident_allocated_upper_bytes']
 need(type(gpu) is int and math.ceil(p['finished_epoch']-p['started_epoch'])+60<=gpu<=120
      and type(cpu) is int and 0<=cpu<=3 and type(output) is int and 64*1024**2<=output<=256*1024**2,
      'FULL_WALL_PAD_CPU_AND_EXACT_SMALL_ALLOCATED_OUTPUT')
 held=l['Core_native_GPU_reservations'][RID]
 need(held['status']=='RESERVED_NOT_SPENT' and held['request_sha256']==REQUEST_SHA
      and RID not in l.get('Core_native_GPU_terminal_settlements',{}),'EXACT_PAIRED_LOCAL_ESCROW')
 g,l=deepcopy(g),deepcopy(l);r=g[c.FIELD][RID]
 effects=dict(schema='PV1_CORE_PREBIND_COMPLETE_ZERO_EFFECTS_V1',request_id=RID,request_sha256=REQUEST_SHA,
              source_sha=SOURCE,actual_model_loads=0,actual_queries=0,actual_physics=0,actual_training_updates=0,unknown_effects=[])
 r.update(status=c.STATES[3],native_exit_status=1,native_start_rejected_before_binding=True,terminal_identity_kind=KIND,
          actual_failed_systemd_InvocationID=p['InvocationID'],actual_failed_systemd_ExecMainPID=p['ExecMainPID'],
          query_attempts=0,model_load_attempts=0,unknown_query_intents=[],reserved_neural_queries=0,
          conservative_GPU_s=gpu,conservative_CPU_s=cpu,terminal_native_sha256=c.digest(p),
          terminal_effect_summary_sha256=c.digest(effects),settled_epoch=now_epoch)
 g['used']['H200_s']+=gpu;g['reserved']['H200_s']-=c.GPU_UPPER;g['CPU_s']+=cpu;g['CPU_reserved_s']-=c.CPU_UPPER
 g['reserved']['neural_queries']-=r['query_limit'];g['storage_reserved_bytes']-=r['additional_output_upper_bytes']
 g['new_storage_current_bytes']+=output;g['new_storage_peak_bytes']=max(g['new_storage_peak_bytes'],g['new_storage_current_bytes'])
 l['GPU_reserved_s']-=c.GPU_UPPER;l['GPU_allocated_s']+=gpu
 for site in ('CoreLab','Hopper'):l['CPU_reserved_s'][site]-=c.CPU_UPPER;l['CPU_s'][site]+=cpu
 l['Core_native_GPU_reserved']['CoreLab_output_bytes']-=r['additional_output_upper_bytes'];l['Core_native_GPU_reserved']['transfer_bytes']-=2*c.MODEL_BYTES
 l['Core_native_GPU_current_storage_bytes']=l.get('Core_native_GPU_current_storage_bytes',0)+output
 receipt=dict(schema='PV1_CORE_PREBIND_KNOWN_ZERO_DUAL_SETTLEMENT_V1',request_id=RID,request_sha256=REQUEST_SHA,
              source_sha=SOURCE,terminal_identity_kind=KIND,native_handle=None,GPU_s=gpu,Core_CPU_s=cpu,
              query_attempts=0,model_load_attempts=0,transfer_accounted_bytes=0,new_resident_allocated_upper_bytes=output,
              native_terminal_sha256=c.digest(p),actual_effects_sha256=c.digest(effects),IO_performed=False)
 l.setdefault('Core_native_GPU_terminal_settlements',{})[RID]=receipt
 l['Core_native_GPU_reservations'][RID].update(status='SETTLED_KNOWN_NATIVE_TERMINAL',terminal_settlement_sha256=c.digest(receipt))
 c.checked(g);c.scope.validate_pair(g,l)
 return dict(proposed_global=g,proposed_LOCAL=l,receipt=receipt,effects=effects,IO_performed=False,new_admission=False)
