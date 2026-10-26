"""GEN1 saved-label binding for the original official CPU exporter.

This module does not register, allocate, copy raw data, run a processor or grant
training admission. Native callers still need closed capture custody and their
own current resource admission. The old PV1 population gate stays unchanged.
"""
from copy import deepcopy

from . import amd_capture as capture, amd_pilot_scope as scope


def exact_card(card, packet_bytes):
    expected = {c['id']: c for i in (0, 1) for c in capture.cards(packet_bytes, i)}
    scope.need(type(card) is dict and card.get('id') in expected
               and card == expected[card['id']], 'EXPORT_EXACT_FIRST_ATTEMPT_CARD')
    return card


def packet(samples, card, packet_bytes):
    """Use the original committed EEF10 labels, with a distinct GEN1 gate."""
    exact_card(card, packet_bytes)
    scope.need(all(s['causal_history']['episode_id'] == card['id'] for s in samples),
               'EXPORT_SAME_NATIVE_CAUSAL_EPISODE')
    from scripts.pi0294.pv1.export import committed_packet
    return committed_packet(samples, card)


def export_case(case, out, processor, source, cosmos, guard, *, packet_bytes, statistics_bytes):
    """Invoke the full original raw reader, codec, loader and collator chain."""
    from scripts.pi0292.durable import read
    from scripts.pi0294.pv1.export import export_committed_case, codec_packet
    from scripts.pi0294.scene_cards import digest
    from pathlib import Path
    import hashlib
    scope.need(type(statistics_bytes) is bytes
               and hashlib.sha256(statistics_bytes).hexdigest() == scope.a.STATS,
               'EXPORT_EXACT_ORIGINAL_R0_BYTES')
    card = exact_card(read(Path(case['source_episode']) / 'CARD.json'), packet_bytes)
    scope.need(case['id'] == card['id'] and case['card_sha256'] == digest(card),
               'EXPORT_CASE_TO_EXACT_CARD')
    row = export_committed_case(case, out, processor, source, cosmos, guard,
        packet_builder=lambda samples, c: packet(samples, c, packet_bytes),
        codec_auditor=codec_packet,
        direction_for_card=lambda c: next(r['layout_stratum'].split('/')[0]
            for r in scope.pilot_plan(packet_bytes)['attempts'] if r['attempt_id'] == c['id']),
        origin='GEN1_ACTUAL_COMMITTED_ORIGINAL_TEACHER_G_TRAIN')
    # The shared reader's TRAIN field is an internal reader compatibility tag.
    # Explicit GEN identity remains authoritative and must reach the bank gate.
    row.update(study_split='G_TRAIN', geometry_id=card['geometry_id'],
        design_hash=card['GEN1_design_hash'], lineage_id=card['GEN1_lineage_id'],
        visual_profile='V0', source_packet_sha256=scope.PACKET_SHA,
        statistics_sha256=scope.a.STATS,
        compatibility_TRAIN_field_is_not_old_PV1_population=True)
    return row


def support_after_export(packet_bytes, capture_rows, export_rows):
    """Require all 12 outcomes, retain negatives, and use actual codec receipts.

    This joins outcomes only; it cannot establish raw custody or native process
    qualification. In particular, Teacher success by itself cannot pass it.
    """
    plan = scope.pilot_plan(packet_bytes)
    expected = {r['attempt_id']: r for r in plan['attempts']}
    scope.need(type(capture_rows) is list and len(capture_rows) == 12
               and len({r['attempt_id'] for r in capture_rows}) == 12
               and {r['attempt_id'] for r in capture_rows} == set(expected),
               'EXPORT_ALL_FIRST_ATTEMPTS_RETAINED')
    scope.need(type(export_rows) is list
               and len({r['id'] for r in export_rows}) == len(export_rows),
               'EXPORT_NO_DUPLICATE_RECEIPT')
    exported = {r['id']: r for r in export_rows}
    scope.need(set(exported) <= set(expected), 'EXPORT_NO_FOREIGN_EPISODE')
    rows = []
    for captured in capture_rows:
        identity = expected[captured['attempt_id']]
        scope.need(captured['unknown_effects'] == [] and captured['terminal'] is True
                   and type(captured['native_safety_failures']) is int
                   and captured['native_safety_failures'] == 0,
                   'EXPORT_KNOWN_SAFE_CAPTURE')
        row = deepcopy(identity)
        row.update(terminal=True, unknown_effects=[], native_safety_failures=0,
            admitted_for_training=False)
        receipt = exported.get(captured['attempt_id'])
        if receipt is not None:
            scope.need(captured['full_task_success'] is True
                       and captured['independent_raw_audit'] == 'PASS_INDEPENDENT_RAW_AUDIT'
                       and receipt['source_raw_manifest_sha256'] == captured['raw_manifest_sha256']
                       and receipt['source_full_audit_sha256'] == captured['audit_sha256'],
                       'EXPORT_ACTUAL_SUCCESS_AND_RAW_IDENTITIES')
            scope.need(all(receipt[k] == identity[k] for k in
                ('task', 'geometry_id', 'design_hash', 'study_split', 'visual_profile'))
                and receipt['source_packet_sha256'] == scope.PACKET_SHA
                and receipt['statistics_sha256'] == scope.a.STATS,
                'EXPORT_SAME_GEOMETRY_TASK_AND_SPLIT')
            codec = receipt['codec']
            scope.need(type(receipt['admitted_for_training']) is bool,
                       'EXPORT_EXPLICIT_ADMISSION_RESULT')
            if receipt['admitted_for_training']:
                scope.need(receipt['status'] == 'PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
                    and receipt['full_task_success'] is True and receipt['independent_raw_admitted'] is True
                    and receipt['windows'] == 601 and codec['statistics_refit'] is False
                    and codec['encode_clip_components'] == 0
                    and codec['native_decode_unchanged'] is True,
                    'EXPORT_ORIGINAL_FULL_CODEC_CHAIN')
                row.update(admitted_for_training=True, full_task_success=True,
                    independent_raw_audit=captured['independent_raw_audit'],
                    codec_status=codec['status'], statistics_sha256=scope.a.STATS,
                    H40_windows=codec['H40_windows_actually_checked'],
                    Teacher_takeover_of_student=False)
        rows.append(row)
    result = scope.training_support(plan, rows)
    result.update(raw_custody_and_native_resource_admission_still_required=True,
                  native_admission=False, new_physics=0, new_training=0)
    return result
