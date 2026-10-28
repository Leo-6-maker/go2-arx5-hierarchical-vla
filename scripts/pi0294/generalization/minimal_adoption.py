"""Exact seven-cap adoption in the original GLOBAL; never a model release.

The first-phase reservation and all native dispatch still require the original
registered owner. Unavailable node-local LOCAL is never replaced by a copy.
This pure proposal changes only one authority marker and revision.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math

from . import minimal_scope as scope
from scripts.pi0294.pv1 import mix_accounting as original

FIELD = 'ARX5_minimal_first512_seven_cap_authority'
SCHEMA = 'ARX5_MINIMAL_FIRST512_EXISTING_GLOBAL_AUTHORITY_V1'
GRANT_SHA = '6a9830b8b328fce861fed927db24f2875d74b973db9eb86774f991de67493572'
DRAFT_SHA = '051761e498774d76a10a5eed087520c5970303d485f065fcb0031ffcfe124ef6'
BEFORE = dict(H200_s=121*3600, PBS_submissions=37, Hopper_CPU_s=213*3600,
              Core_CPU_s=320*3600, transfer_bytes=332*scope.GIB,
              Hopper_storage_bytes=1176*scope.GIB, Core_storage_bytes=477*scope.GIB)
CAPS = dict(H200_s=147*3600, PBS_submissions=39, Hopper_CPU_s=265*3600,
            Core_CPU_s=380*3600, transfer_bytes=455*scope.GIB,
            Hopper_storage_bytes=1399*scope.GIB, Core_storage_bytes=550*scope.GIB)


def raw(v):
    return (json.dumps(v, indent=2, allow_nan=False) + '\n').encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def need(ok, reason):
    if not ok:
        raise PermissionError('ARX5_MINIMAL_ADOPTION_' + reason)


def grant(value):
    need(type(value) is bytes and sha(value) == GRANT_SHA, 'EXACT_DIRECT_HUMAN_GRANT_BYTES')
    g = json.loads(value)
    need(g['schema'] == 'ARX5_MINIMAL_FIRST512_SEVEN_CAP_DIRECT_HUMAN_GRANT_V1'
         and g['human_answer'] == '批准仅这七项上限调整'
         and g['scope'] == scope.SCOPE and g['draft_sha256'] == DRAFT_SHA
         and g['caps'] == {k:dict(before=BEFORE[k], after=v) for k,v in CAPS.items()}
         and g['original_matrix3_and_final20GPUh40GiB_protected'] is True
         and g['old_C50_D25_GEN1_Evals_and_costs_retained'] is True
         and g['other_caps_unchanged'] is True and g['clocks_not_reset'] is True
         and g['Core4080S_original_measurement_qualification_required_before_full_training'] is True
         and g['conditional1024_included'] is False
         and g['new_door_training_or_LoRA_or_collection_included'] is False
         and g['maximum_extra_zero_effect_engineering_PBS'] == 1
         and g['full_native_admission_by_this_document'] is False,
         'ONLY_EXACT_SEVEN_CAPS_AND_ORIGINAL_SCIENTIFIC_BOUNDARIES')
    return g


def validate(g):
    if FIELD not in g:
        return None
    m = g[FIELD]
    h = grant(base64.b64decode(m['grant_base64'], validate=True))
    prior = original.scope.validated(g)
    need(m['schema'] == SCHEMA and m['scope'] == scope.SCOPE
         and m['grant_sha256'] == GRANT_SHA and m['caps'] == CAPS
         and m['before_caps'] == BEFORE
         and all(m[k] == h[k] == prior[k] for k in
                 ('T_CD', 'stop_new_training_epoch', 'deadline_epoch'))
         and type(m['adopted_epoch']) in (int,float) and math.isfinite(m['adopted_epoch'])
         and prior['T_CD'] <= m['adopted_epoch'] < prior['stop_new_training_epoch']
         and m['actual_LOCAL_written'] is False
         and m['LOCAL_sync'] == 'PENDING_CANONICAL_4080_NODE_LOCAL_NO_COPY'
         and m['native_admission'] is False
         and m['package_reservation_state'] == 'NOT_YET_RESERVED_NO_NATIVE_DISPATCH'
         and m['Core_measurement_gate_for_full_training_preserved'] is True,
         'EXACT_ADOPTED_CAPS_CLOCK_HISTORICAL_LOCAL_AND_NO_DISPATCH')
    for k in ('before_GLOBAL_sha256','historical_LOCAL_sha256','full_before_future_sha256'):
        need(type(m[k]) is str and len(m[k]) == 64
             and all(c in '0123456789abcdef' for c in m[k]), 'EXACT_SOURCE_BYTE_IDENTITY')
    return m


def effective_caps(g, caps):
    """Read only the exact new authority overlay; legacy markers stay intact."""
    out = dict(caps)
    m = validate(g)
    if m is not None:
        from . import minimal_gpu_repair as repair
        extra=repair.validate(g)
        from . import minimal_followup_scope as followup
        followup_marker=followup.validate(g)
        from . import minimal_entry_continuation_scope as entry
        entry_marker=entry.validate(g)
        from . import minimal_partial_completion_scope as completion
        completion_marker=completion.validate(g)
        need(all(out[k] in (BEFORE[k], CAPS[k], repair.AFTER if extra and k=='H200_s' else CAPS[k],
            followup.AFTER if followup_marker and k=='PBS_submissions'else CAPS[k],
            entry.AFTER if entry_marker and k=='PBS_submissions'else CAPS[k],
            completion.AFTER if completion_marker and k=='PBS_submissions'else CAPS[k]) for k in CAPS), 'ONLY_OLD_OR_EXACT_NEW_ADOPTED_CAPS')
        out.update(m['caps'])
        if extra is not None:out['H200_s']=repair.AFTER
        if followup_marker is not None:out['PBS_submissions']=followup.AFTER
        if entry_marker is not None:out['PBS_submissions']=entry.AFTER
        if completion_marker is not None:out['PBS_submissions']=completion.AFTER
    return out


def global_limits(g, caps):
    """Versioned aggregate ceilings for the existing ledger's legacy readers.

    Scientific per-request limits and old authority bytes are unchanged. A
    complete validated new authority is necessary even for a partial reader.
    """
    out = dict(caps)
    m = validate(g)
    if m is not None:
        from . import minimal_gpu_repair as repair
        extra=repair.validate(g)
        from . import minimal_followup_scope as followup
        followup_marker=followup.validate(g)
        from . import minimal_entry_continuation_scope as entry
        entry_marker=entry.validate(g)
        from . import minimal_partial_completion_scope as completion
        completion_marker=completion.validate(g)
        for k in set(out) & set(CAPS):
            ceiling=repair.AFTER if extra and k=='H200_s' else CAPS[k]
            if k=='PBS_submissions'and followup_marker is not None:ceiling=followup.AFTER
            if k=='PBS_submissions'and entry_marker is not None:ceiling=entry.AFTER
            if k=='PBS_submissions'and completion_marker is not None:ceiling=completion.AFTER
            need(type(out[k]) in (int,float) and math.isfinite(out[k])
                 and 0 <= out[k] <= ceiling, 'KNOWN_GLOBAL_CEILING_INPUT:' + k)
            out[k] = ceiling
    return out


def propose(global_bytes, historical_LOCAL_bytes, grant_bytes, *, now_epoch):
    approved = grant(grant_bytes)
    need(type(global_bytes) is bytes and type(historical_LOCAL_bytes) is bytes,
         'COMPLETE_CAPTURED_LEDGER_BYTES_REQUIRED')
    g = json.loads(global_bytes)
    need(FIELD not in g, 'ONCE_NO_READOPTION_OR_SILENT_OVERWRITE')
    original.validate_global(g)
    need(g['pending_updates'] == g['unknown_effects'] == []
         and all(r['status'] == 'NATIVE_TERMINAL' for r in g['PBS_submissions'])
         and all(r['status'] == 'NATIVE_TERMINAL' for r in g.get(original.CORE_ROWS, {}).values()),
         'NO_ACTIVE_OR_UNKNOWN_GPU_EFFECTS')
    prior = original.scope.validated(g)
    need(type(now_epoch) in (int,float) and math.isfinite(now_epoch)
         and prior['T_CD'] <= now_epoch < prior['stop_new_training_epoch']
         and all(prior[k] == approved[k] for k in
                 ('T_CD','stop_new_training_epoch','deadline_epoch')),
         'ORIGINAL_WINDOW_NOT_RESET_BY_NEW_AUTHORITY')
    historical = g['GEN1_AMD_layout_CORR1024_authority']['historical_LOCAL_sha256']
    need(sha(historical_LOCAL_bytes) == historical, 'EXACT_HISTORICAL_LOCAL_NOT_FRESH_OR_AMD_COPY')
    proof = scope.first512_forecast(global_bytes, historical_LOCAL_bytes)
    for key,row in proof['dimensions'].items():
        need(row['cap'] == BEFORE.get(key, row['cap'])
             and row['projected'] <= CAPS.get(key, row['cap']), 'FULL_REMAINING_PACKAGE_FITS:' + key)
    m = dict(schema=SCHEMA, scope=scope.SCOPE, grant_sha256=GRANT_SHA,
        grant_base64=base64.b64encode(grant_bytes).decode(), caps=deepcopy(CAPS), before_caps=deepcopy(BEFORE),
        T_CD=prior['T_CD'], stop_new_training_epoch=prior['stop_new_training_epoch'],
        deadline_epoch=prior['deadline_epoch'], adopted_epoch=now_epoch,
        before_GLOBAL_sha256=sha(global_bytes), historical_LOCAL_sha256=sha(historical_LOCAL_bytes),
        full_before_future_sha256=sha(raw(proof)),
        LOCAL_sync='PENDING_CANONICAL_4080_NODE_LOCAL_NO_COPY', actual_LOCAL_written=False,
        package_reservation_state='NOT_YET_RESERVED_NO_NATIVE_DISPATCH',
        Core_measurement_gate_for_full_training_preserved=True, native_admission=False)
    out = deepcopy(g); out[FIELD] = m; out['revision'] += 1
    validate(out)
    need(all(out[k] == v for k,v in g.items() if k != 'revision'),
         'ALL_OLD_COSTS_RESERVATIONS_HISTORY_AND_LEASES_PRESERVED')
    after = scope.first512_forecast(raw(out), historical_LOCAL_bytes)
    need(after['cap_changes_required'] == {}, 'WHOLE_FIRST_PHASE_AND_OLD_WORK_PAYABLE_AFTER_CAP_ADOPTION')
    return dict(proposed_global=out, marker=m, full_future_proof=after,
                actual_LOCAL_changed=False, native_admission=False,
                new_requests=0, new_PBS=0, new_GPU=0)
