"""Cold Core CPU classification for the two already authorized Eval panels.

GPU-site CPU remains its original projection. The same work also consumes CPU
on physical Core; reserve that dimension from existing cap headroom once.
Original package declaration, caps, clocks and paid history stay unchanged.
"""
from copy import deepcopy
import json
import re

from . import mix_accounting as a, successor_cd as s

FIELD='V5_CD_existing_Core_GPU_CPU_projection'
PER_PANEL=80000


def need(ok,reason):
    if not ok:raise PermissionError('V5_CD_CORE_CPU_PROJECTION_'+reason)


def validate(g,l=None):
    marker=g.get(FIELD)
    need(type(marker)is dict and marker.get('schema')=='V5_CD_EXISTING_CORE_GPU_CPU_PROJECTION_V1'
         and marker.get('task')==s.TASK and marker.get('scope_grant_sha256')==s.GRANT_SHA
         and marker.get('per_panel_Core_CPU_upper_s')==PER_PANEL
         and type(marker['per_panel_Core_CPU_upper_s'])is int
         and marker.get('additional_Core_CPU_reserved_s')==2*PER_PANEL
         and type(marker['additional_Core_CPU_reserved_s'])is int
         and marker.get('panels')==['EVAL:C50:N294101','EVAL:D25:N294101']
         and marker.get('caps_changed')=={} and marker.get('new_scientific_work')is False
         and marker.get('original_GPU_site_CPU_retained')is True,
         'ACTUAL_ONCE_CLASSIFICATION_NOT_NEW_CAP_OR_GPU_COST')
    need(g[a.PACKAGE].get(FIELD)==marker,'SAME_EXISTING_ESCROW_MARKER')
    need(all(type(marker.get(k))is str and re.fullmatch('[0-9a-f]{64}',marker[k])
             for k in ('before_GLOBAL_sha256','before_LOCAL_sha256'))
         and type(marker.get('observed_epoch'))in(int,float), 'ACTUAL_DUAL_BEFORE_IDENTITY')
    a.number(marker['observed_epoch'])
    if l is not None:need(l.get(FIELD)==marker and l[a.PACKAGE]==g[a.PACKAGE], 'EXACT_DUAL_ADOPTION')
    return marker


def propose(global_bytes,local_bytes,*,Core_storage_upper_bytes,local_storage_upper_bytes,now_epoch):
    need(type(global_bytes)is bytes and type(local_bytes)is bytes,'FULL_RAW_EXISTING_DUAL_BYTES')
    from . import mix_core_accounting as c
    g,l=json.loads(global_bytes),json.loads(local_bytes)
    c.checked(g);c.paired(g,l);p=a.package_view(g);now=a.number(now_epoch)
    need(FIELD not in g and FIELD not in l and FIELD not in p,'ONE_NEW_CLASSIFICATION_NEVER_REPLAY')
    need(p['completed_stages']==['TRAIN:C50:1024'] and not g.get(a.CORE_ROWS)
         and all(r['status']in a.TERMINAL for r in g['PBS_submissions'])
         and not any(r['status']!='NATIVE_TERMINAL'for r in g.get('Core_native_GPU_allocations',{}).values())
         and s.validated(g)['T_CD']<=now<s.validated(g)['deadline_epoch']-2*28860-22560,
         'ONLY_INACTIVE_CLOSED_C50_BEFORE_BOTH_ORIGINAL_EVALS')
    marker=dict(schema='V5_CD_EXISTING_CORE_GPU_CPU_PROJECTION_V1',task=s.TASK,scope_grant_sha256=s.GRANT_SHA,
        per_panel_Core_CPU_upper_s=PER_PANEL,additional_Core_CPU_reserved_s=2*PER_PANEL,
        panels=['EVAL:C50:N294101','EVAL:D25:N294101'],caps_changed={},new_scientific_work=False,
        original_GPU_site_CPU_retained=True,before_GLOBAL_sha256=a.sha(global_bytes),
        before_LOCAL_sha256=a.sha(local_bytes),observed_epoch=now)
    v,w=deepcopy(g),deepcopy(l)
    for value in (v,w):
        value[FIELD]=deepcopy(marker);value[a.PACKAGE][FIELD]=deepcopy(marker)
        value[a.PACKAGE]['remaining_resources']['Core_CPU_s']+=2*PER_PANEL
    w['CPU_reserved_s']['CoreLab']+=2*PER_PANEL
    validate(v,w);c.checked(v);c.paired(v,w)
    proof=s.recompute(v,w,Core_storage_upper_bytes=Core_storage_upper_bytes,
        local_storage_upper_bytes=local_storage_upper_bytes)
    need(proof['shortfalls']=={},'ALL_REMAINING_CAPS_MATRIX_AND_FINAL:'+str(proof['shortfalls']))
    return dict(proposed_global=v,proposed_LOCAL=w,receipt=marker,full_remaining_proof=proof,
        IO_performed=False,original_onceIO_required=True,native_model_or_physics_admission=False)
