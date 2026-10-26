"""Cold site continuation of the two original C/D panels, in the PV1 ledger.

No scheduler, second budget or Core ledger is created here. Proposals require
both actual closed training endpoints and retain the unavailable LOCAL bytes as
an explicitly historical witness. Native entry and release remain separate.
"""
from copy import deepcopy
import base64
import json
import os
from pathlib import Path

from . import mix_accounting as a, mix_vanda_first as first, successor_cd as s
from . import mix_core_premodel_repair as repair, mix_stage_handoff as handoff

FIELD = 'V5_CD_Vanda_colocated_original_panels'
SCHEMA = 'V5_CD_VANDA_COLOCATED_ORIGINAL_PANELS_V1'
AUTH_SHA = '7d546f2bf1ae94f04b3c70ea15b8473278a1b3f4f4df16ce917fd3ca3886093e'
GIB = 1024**3
MIB = 1024**2
PHYSICAL_OUTPUT = 26*768*MIB
QUERY_OUTPUT = 26*80*8*MIB
OTHER_OUTPUT = 3*GIB
COHOST_OUTPUT = PHYSICAL_OUTPUT+QUERY_OUTPUT+OTHER_OUTPUT
EXTRA_SITE_CPU = 189000
EXTRA_PBS = 3  # Two original panels plus one bounded zero-integration CPU job.
QUALIFICATION_CPU = 960
QUALIFICATION_CONSTRUCTORS = 2


def need(ok, why):
    if not ok:
        raise PermissionError('V5_VANDA_COHOST_'+why)


def storage_contract():
    """An execution bound within the original paid/reserved neural IO upper.

    The original 768MiB per physical attempt is retained. Additional per-query
    files (three observation/request copies plus action/ACK/receipts) share an
    8MiB envelope. Qualification, logs and all other new artifacts share 3GiB.
    These are engineering resource fences, not filters on saved raw evidence.
    Any overflow stops with all files retained; it never truncates/deletes raw.
    Original model tensors are referenced read-only, never copied into output.
    """
    from .mix_core_accounting import remaining_stage
    original = remaining_stage()['Hopper_storage_bytes']
    need(COHOST_OUTPUT < original, 'ALL_COLOCATED_ARTIFACTS_INSIDE_ORIGINAL_GROSS_UPPER')
    return dict(schema='V5_CD_COLOCATED_RETAINED_OUTPUT_PARTITION_V1',
        attempts=26, physical_per_attempt_bytes=768*MIB,
        query_calls=26*80, additional_per_query_bytes=8*MIB,
        other_artifacts_bytes=OTHER_OUTPUT, total_bytes=COHOST_OUTPUT,
        original_gross_upper_bytes=original, model_payload_copies=0,
        old_fees_or_peak_refunded=False, raw_artifacts_deleted_or_truncated=False,
        overflow_action='STOP_PRESERVE_ALL_ENGINEERING_INCOMPLETE')


def tree_bytes(root, *, exclude=()):
    """Count unique inodes without following a link or crossing a mount.

    Model inputs must be outside these output roots; arbitrary exclusions are
    forbidden by the entry validator, which alone supplies the sealed roles.
    This guard does not write, prune or change any source/input inode metadata.
    """
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'REAL_OUTPUT_ROOT')
    excluded = {Path(x) for x in exclude}
    need(all(p.is_absolute() and p.is_relative_to(root) for p in excluded), 'EXCLUSIONS_WITHIN_BOUND_ROOT')
    seen = set(); total = 0; allocated = 0; device = root.stat().st_dev
    for current, directories, files in os.walk(root, followlinks=False):
        for name in list(directories):
            p = Path(current)/name
            need(not p.is_symlink() and p.stat().st_dev == device, 'NO_OUTPUT_LINK_OR_MOUNT_ESCAPE')
            if p in excluded:
                directories.remove(name)
        for name in files:
            p = Path(current)/name
            if p in excluded:
                continue
            st = p.lstat()
            import stat
            need(stat.S_ISREG(st.st_mode) and st.st_dev == device, 'REGULAR_SAME_FILESYSTEM_ARTIFACT')
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key); total += st.st_size; allocated += st.st_blocks*512
    return dict(logical_bytes=total, allocated_bytes=allocated, unique_inodes=len(seen))


def enforce_output(root, maximum):
    need(type(maximum) is int and maximum > 0, 'EXACT_POSITIVE_OUTPUT_BOUND')
    measured = tree_bytes(root)
    need(max(measured['logical_bytes'], measured['allocated_bytes']) <= maximum,
         'RETAINED_OUTPUT_UPPER_EXCEEDED_PRESERVE_ALL')
    return measured


def endpoint(global_bytes, capture):
    need(capture.get('schema') == 'V5_CD_REAL_TRAINING_TERMINAL_BYTES_V1', 'ACTUAL_TERMINAL_CAPTURE')
    terminal = {k:base64.b64decode(v, validate=True)
                for k, v in capture['terminal_capture_base64'].items()}
    # Rebind real terminal bytes to the actual current GLOBAL, never a stale
    # handoff hash or caller's prepared PASS flag.
    return handoff.closed_endpoint(global_bytes=global_bytes, arm=capture['arm'], **terminal)


def validated(g):
    marker = g.get(FIELD)
    if marker is None:
        return None
    clock = s.validated(g)
    need(marker['schema'] == SCHEMA and marker['authority_sha256'] == AUTH_SHA
         and marker['scope_grant_sha256'] == s.GRANT_SHA
         and marker['T_CD'] == clock['T_CD']
         and marker['deadline_epoch'] == clock['deadline_epoch']
         and marker['effective_eval_order'] == ['C50-DEV-N294101', 'D25-DEV-N294101']
         and marker['storage_contract'] == storage_contract()
         and marker['extra_site_CPU_s'] == EXTRA_SITE_CPU
         and marker['extra_PBS'] == EXTRA_PBS
         and marker['confirmed_LOCAL_sha256'] == first.LOCAL_SHA
         and marker['LOCAL_sync'] == 'PENDING'
         and marker['Core_dispatch_permitted'] is False
         and marker['historical_LOCAL_is_native_authority'] is False
         and marker['old_UNKNOWN_preserved'] is True,
         'EXACT_SITE_SCOPE_STORAGE_COST_AND_DEFERRED_LOCAL')
    need(set(marker['closed_endpoints']) == {'C50', 'D25'}, 'BOTH_REAL_CLOSED_ARMS')
    for arm, proof in marker['closed_endpoints'].items():
        end = g[a.PACKAGE]['endpoints'][arm]
        need(proof['arm'] == arm and proof['checkpoint_manifest_sha256'] == end['checkpoint_manifest_sha256']
             and proof['global_step'] == 1024 and proof['request_id'] == end['request_id']
             and proof['independent_audit_sha256'] == end['audit_sha256'], 'EXACT_RETAINED_ENDPOINT:'+arm)
    need(marker['C50_paid_IO_marker_sha256'] == a.sha(a.raw(repair.validated(g))), 'ONLY_EXACT_C50_PAID_IO')
    return marker


def propose(global_bytes, historical_LOCAL_bytes, *, authority_bytes, captures, now_epoch):
    g = json.loads(global_bytes); local = json.loads(historical_LOCAL_bytes)
    a.validate_global(g)
    need(FIELD not in g, 'ONCE_SITE_CLASSIFICATION_NO_REPLAY')
    need(a.sha(authority_bytes) == AUTH_SHA, 'DIRECT_HUMAN_CONTINUATION_BYTES')
    need(a.sha(historical_LOCAL_bytes) == first.LOCAL_SHA, 'ORIGINAL_CONFIRMED_LOCAL_BYTES_NOT_FRESH')
    need(first.validated(g)['LOCAL_sync'] == 'PENDING'
         and g[a.PACKAGE]['completed_stages'] == ['TRAIN:C50:1024', 'TRAIN:D25:1024']
         and not g['unknown_effects'] and not g['pending_updates']
         and all(r['status'] in a.TERMINAL for r in g['PBS_submissions'])
         and all(r['status'] == 'NATIVE_TERMINAL' for r in g.get(a.CORE_ROWS, {}).values()),
         'BOTH_TRAININGS_CLOSED_NO_ACTIVE_GPU_OR_UNKNOWN_EFFECT')
    need(set(captures) == {'C50', 'D25'} and all(captures[k]['arm'] == k for k in captures), 'EXACT_TWO_TERMINAL_CAPTURE_ROLES')
    clock = s.validated(g)
    need(clock['T_CD'] <= a.number(now_epoch) < clock['deadline_epoch']-2*28860,
         'BOTH_COMPLETE_PANELS_AND_CLOSEOUT_FIT_ORIGINAL_WINDOW')
    proof = {k:endpoint(global_bytes, v) for k, v in captures.items()}
    marker = dict(schema=SCHEMA, authority_sha256=AUTH_SHA, scope_grant_sha256=s.GRANT_SHA,
        before_GLOBAL_sha256=a.sha(global_bytes), confirmed_LOCAL_sha256=first.LOCAL_SHA,
        LOCAL_sync='PENDING', historical_LOCAL_is_native_authority=False,
        Core_dispatch_permitted=False, old_UNKNOWN_preserved=True,
        T_CD=clock['T_CD'], deadline_epoch=clock['deadline_epoch'], adopted_epoch=now_epoch,
        effective_eval_order=['C50-DEV-N294101', 'D25-DEV-N294101'],
        closed_endpoints=proof, storage_contract=storage_contract(),
        C50_paid_IO_marker_sha256=a.sha(a.raw(repair.validated(g))),
        extra_site_CPU_s=EXTRA_SITE_CPU, extra_PBS=EXTRA_PBS,
        original_Core_CPU_and_output_holds_retained=True, no_budget_cap_or_historical_fee_change=True)
    v = deepcopy(g); v[FIELD] = marker
    # Keep every old Core hold conservatively; add physical-site CPU from
    # actual uncommitted site headroom. Storage needs no new credit: same-host
    # input references and all output partitions fit the ORIGINAL gross upper.
    v[a.PACKAGE]['remaining_resources']['Hopper_CPU_s'] += EXTRA_SITE_CPU
    v[a.PACKAGE]['remaining_resources']['PBS_submissions'] += EXTRA_PBS
    v['CPU_reserved_s'] += EXTRA_SITE_CPU
    validated(v); a.validate_global(v)
    full = s.recompute(v, local, Core_storage_upper_bytes=first.CORE_STORAGE,
                       local_storage_upper_bytes=first.LOCAL_STORAGE)
    need(not full['shortfalls'], 'ALL_REMAINING_MATRIX_AND_FINAL_PAYABLE:'+str(full['shortfalls']))
    return dict(proposed_global=v, marker=marker, full_remaining_proof=full,
                actual_LOCAL_changed=False, native_admission=False, native_IO_performed=False)
