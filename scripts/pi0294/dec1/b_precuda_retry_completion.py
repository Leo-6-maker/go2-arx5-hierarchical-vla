"""Exact actual1481574 terminal binding; no native work on import.

Only its true F/Exit0, natural owner OS exit0, three full15-member states,
original12 probes and complete actual query census can satisfy this contract.
"""
from .b_recovery2_completion_contract import CompletionContract,need
from . import b_precuda_retry as authority,b_vanda_recovery2 as recovery
SOURCE='36925d272095d000ba1ad3ce5e9cdc44373fdce6'
RUN='vap1-a40-b-recovery2-retry1-01'
JOB='1481574.stdct-mgmt-02'
REQUEST='310edd77-f972-4dbd-baa8-4c2c674c08ea'
REQUEST_SHA='e02b55d3a0946ef1453c3f38e5a8f5c13655c4b6cd225349d49ce7077e9131a2'
NONCE='d476a779-4ca2-4442-8200-15924a32921f'
GPU_MANIFEST='e424d578434062b8e1c8e31c81d2635584d83f96da0ce3ec5e1736d1e331ff47'
CPU_MANIFEST='ebc7e699be700bf5912725f579d6923777636c529896853b2aabfb709289f98d'
OWNER_PID=372546
OWNER_START='28323899'
WRAPPER_PID=372539
WRAPPER_START='28323898'

class _ActualAttempt(CompletionContract):
    def expected(self,value):
        row=super().expected(value)
        need(value.get('job_id')==JOB and value.get('request_sha256')==REQUEST_SHA
             and row.get('request_id')==REQUEST and row.get('submission_nonce')==NONCE
             and value['registration'].get('manifest_sha256')==GPU_MANIFEST
             and value.get('control_manifest_sha256')==CPU_MANIFEST
             and row.get('sealed_inputs',{}).get(authority.PLAN)==authority.GRANT_SHA,
             'EXACT_ACTUAL1481574_REGISTERED_GPU_CPU_REQUEST_NONCE_AND_SEALED_GRANT')
        return row

    def owner(self,packet,value,*,now_epoch):
        super().owner(packet,value,now_epoch=now_epoch)
        need(packet['owner_process'].get('pid')==OWNER_PID
             and packet['owner_process'].get('proc_start_ticks')==OWNER_START
             and packet['launch'].get('pid')==WRAPPER_PID
             and packet['launch'].get('proc_start_ticks')==WRAPPER_START,
             'EXACT_ORIGINAL_BACKGROUND_OWNER_AND_FOREGROUND_WRAPPER_INSTANCES')

_contract=_ActualAttempt(source=SOURCE,run=RUN,authority=authority)
audit=_contract.audit
effects=_contract.effects
expected=_contract.expected
native=_contract.native
origin=_contract.origin
owner=_contract.owner
query_provenance=_contract.query_provenance
settled=_contract.settled
