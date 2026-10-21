"""Same saved raw through fixed R0/R1, with actual official dual-camera readers."""
import copy
import os
from pathlib import Path
import resource
import time

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import cpu_guard, validate_samples
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0292.train_adapter import build_processor
from scripts.pi0294.cr1.numeric import serial
from scripts.pi0294.cr1.scope import install_import_guard
from scripts.pi0294.data_cov1.codec import audit as codec_audit
from scripts.pi0294.data_cov1.export import event_index, join_codec_event
from scripts.pi0294.e1.artifacts import atomic_npz
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.native_dataset import NativeDataset
from scripts.pi0294.pv1.export import batch_inventory, committed_packet
from scripts.pi0294.scene_cards import digest, task_card
from .plan import validate_group
from .runtime import bind_cpu
from .scope import sealed_file, validate


def packet(samples, card, *, profile=None):
    from .runtime import context
    _, _, _, group_validator, _ = context(profile)
    group_validator(card)
    prefix = 'PI0294-VAP1_DATA-TRAIN-' if profile == 'VAP1' else 'PI0294-CR2_FIXED_R1-TRAIN-'
    if (card['nominal_clearance_m'] != .004 or card['task'] not in ('T1', 'T2')
            or card['instruction'] != task_card(card, card['task'])['instruction']
            or not card['id'].startswith(prefix)
            or any(s['causal_history']['episode_id'] != card['id'] for s in samples)):
        raise PermissionError('CR2_EXACT_OBSERVED_LANGUAGE_AND_COMMITTED_TRAIN_LABELS')
    return committed_packet(samples, card)


def readers(out, case, numeric, processor, statistics_sha, source, cosmos, guard):
    row = dict(id=case['id'], split='TRAIN', task=case['task'], scene_group=case['scene_group'],
        windows=601, file=str(numeric.relative_to(out)), sha256=sha(numeric), raw_root='EXPLICIT_PRIVATE_BINDING_REQUIRED')
    write(out / 'DATASET_MANIFEST.json', dict(domain='FIXED_BASE_ARX5_TASK_SIM', statistics_sha256=statistics_sha,
        purpose='CR2_FIXED_VERSION_SAVED_DATA_READER_NOT_READY_BANK', episodes=[row]))
    dataset = NativeDataset(out, source, cosmos, for_optimization=True,
                            raw_bindings={case['id']: case['source_episode']})
    dataset.processor = processor
    batches = []; processed = []
    for anchor in (0, 80, 160, 320, 480, 600):
        guard(); sample = dataset[anchor]['processed']; processed.append(sample)
        batches.append(batch_inventory(processor.collator([sample]), 1))
    repeat = batch_inventory(processor.collator([dataset[0]['processed']]), 1)
    if repeat != batches[0]:
        raise ValueError('CR2_OFFICIAL_RGB_READER_DETERMINISTIC_REPEAT_REQUIRED')
    return dict(actual_anchors=[0, 80, 160, 320, 480, 600], actual_dual_camera_frames=True,
                mask_H40_EEF10_zero_padding=True, deterministic_repeat=True,
                statistics_sha256=statistics_sha, batch_sha256=[digest(v) for v in batches]), processed[0]


def run(root, *, profile=None):
    from .runtime import context
    validate, _, _, _, _ = context(profile)
    r = Path(root); lock = OwnerLock(r / 'CR2_WORKER.lock')
    m, inputs = validate(r, 'EXPORT'); work = read(r / 'plans/WORK_PLAN.private.json')
    ledger = read(r / 'RESOURCE_LEDGER.json'); assert ledger['status'] == 'RESERVED' and not ledger['unknown_effects']
    began = time.process_time(); before = resource.getrusage(resource.RUSAGE_CHILDREN); rows = []; failure = None
    bind_cpu(m)
    def guard():
        if (time.time() >= m['limits']['operation_deadline_epoch'] or time.process_time() - began >= m['limits']['CPU_s'] - 30
                or any((p / 'STOP.json').exists() for p in (r, r.parent.parent))):
            raise TimeoutError('CR2_PAIRED_READER_FINITE_CPU_WALL_OR_STOP')
    try:
        install_import_guard(); cpu_guard()
        processors = {}; statistics = {}
        for version in ('R0', 'R1'):
            statistics[version] = read(r / 'plans' / (version + '_statistics.json'))
            processors[version] = build_processor(inputs['official_source'], inputs['cosmos'], statistics[version])
        params = {v: serial(p.state_action_processor.norm_params) for v, p in processors.items()}
        reference = read(r / 'plans/FIXED_EFFECTIVE_PARAMS.private.json')
        if (params['R0'] != reference['original_effective_params']
                or params['R1'] != reference['candidate_effective_params']):
            raise ValueError('CR2_ACTUAL_PROCESSORS_MUST_MATCH_BOTH_FIXED_CR1_EFFECTIVE_PARAMETER_IDENTITIES')
        expected = copy.deepcopy(params['R0'])
        for key in ('min', 'max'):
            expected['new_embodiment']['action']['eef_9d'][key] = statistics['R1']['new_embodiment']['relative_action']['eef_9d'][key]
        if params['R1'] != expected:
            raise ValueError('CR2_EFFECTIVE_FIXED_R1_ONLY_RELATIVE_EEF9_MINMAX')
        put(r / 'evidence/ACTUAL_EFFECTIVE_PARAMS.json', dict(R0=params['R0'], R1=params['R1'], new_range_fits=0))
        first = {}; first_ids = {}; mixed = {}
        ledger['status'] = 'RUNNING'; write(r / 'RESOURCE_LEDGER.json', ledger)
        for case in work['cases']:
            guard(); folder = Path(case['source_episode']); card = read(folder / 'CARD.json')
            binding = read(folder.parent.parent / 'OUTPUT_BINDING.private.json')
            if (read(binding['lease'])['status'] != 'CLOSED' or card != case['card']
                    or sha(folder / 'FILE_MANIFEST.json') != case['raw_manifest_sha256']
                    or read(folder.parent.parent / 'RESOURCE_LEDGER.json')['unknown_effects']):
                raise PermissionError('CR2_ONLY_EXACT_CLOSED_CAPTURE_RAW_INPUTS')
            result = read(folder / 'RESULT.json'); score = read(folder / 'SCORE.json') if (folder / 'SCORE.json').exists() else {}
            s1 = read(folder / 'S1_RESULT.json') if (folder / 'S1_RESULT.json').exists() else {}
            summary = dict(id=case['id'], scene_group=card['group_id'], task=card['task'],
                appearance_profile=card['appearance_profile'], direction=card['direction'], distance_bin=card['distance_bin'],
                task_success=bool(score.get('persistent_success')), native_faults=s1.get('live_native_failures', 0),
                unknown_effects=[], R0_PASS_to_R1_FAIL=False, admitted=False, evidence_qualified=False,
                capture_status=result['status'], source_raw_manifest_sha256=case['raw_manifest_sha256'])
            if result['status'] != 'VALID_COMPLETE' or not (folder / 'AUDIT.json').exists():
                summary.update(status='PRESERVED_NEGATIVE_NOT_SUCCESS_DATA', paired_codec='NOT_APPLICABLE_INCOMPLETE_RAW')
                rows.append(summary); continue
            audit = read(folder / 'AUDIT.json')
            if sha(folder / 'AUDIT.json') != case['audit_sha256'] or summary['native_faults']:
                raise ValueError('CR2_FULL_RAW_AUDIT_IDENTITY_OR_NATIVE_FAULT')
            for member in read(folder / 'FILE_MANIFEST.json'):
                guard(); path = folder / member['path']
                if (Path(member['path']).is_absolute() or '..' in Path(member['path']).parts or path.is_symlink()
                        or not path.resolve().is_relative_to(folder.resolve())
                        or path.stat().st_size != member['bytes'] or sha(path) != member['sha256']):
                    raise ValueError('CR2_RAW_MEMBER_IDENTITY_CHANGED_OR_ESCAPE')
            samples = read(folder / 'samples.json'); validate_samples(samples, verify_images=folder)
            arrays = packet(samples, card, profile=profile)
            with np.load(folder / 'physics.npz', allow_pickle=False) as z:
                physics = {k: z[k].copy() for k in z.files}
            events = event_index(physics, card); events['original_score'] = score
            put(r / 'evidence' / (case['id'] + '-events.json'), events)
            versions = {}; reader_proofs = {}
            for version, processor in processors.items():
                guard(); out = r / 'datasets' / version / case['id']; out.mkdir(parents=True)
                os.link(r / 'plans' / (version + '_statistics.json'), out / 'statistics.json')
                stats_sha = sha(out / 'statistics.json')
                codec = codec_audit(arrays, processor, guard); join_codec_event(codec, folder, card)
                codec.update(actual_statistics_sha256=stats_sha, statistics_version=version,
                    legacy_PASS_string_refers_only_to_original_numeric_gates_not_statistics_identity=True)
                versions[version] = codec
                numeric = out / 'numeric' / (case['id'] + '.npz'); numeric.parent.mkdir()
                atomic_npz(numeric, **arrays); numeric.chmod(0o400)
                proof, processed = readers(out, case, numeric, processor, stats_sha,
                                            inputs['official_source'], inputs['cosmos'], guard)
                reader_proofs[version] = proof
                if version not in first:
                    first[version], first_ids[version] = processed, case['id']
                elif version not in mixed:
                    mixed[version] = dict(parent_ids=[first_ids[version], case['id']],
                        statistics_sha256=stats_sha,
                        batch_sha256=digest(batch_inventory(processor.collator([first[version], processed]), 2)))
                put(out / 'VERSION_READER_RECEIPT.json', dict(id=case['id'], version=version,
                    statistics_sha256=stats_sha, numeric_sha256=sha(numeric), codec=codec, official_reader=proof))
            passed = {v: c['status'] == 'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP' for v, c in versions.items()}
            summary.update(status='PAIRED_FIXED_R0_R1_SAVED_DATA_COMPLETE',
                evidence_qualified=audit['status'] == 'PASS_INDEPENDENT_RAW_AUDIT',
                R0=versions['R0'], R1=versions['R1'], official_reader=reader_proofs,
                transition=('PASS' if passed['R0'] else 'FAIL') + '->' + ('PASS' if passed['R1'] else 'FAIL'),
                R0_PASS_to_R1_FAIL=passed['R0'] and not passed['R1'],
                admitted=bool(summary['task_success'] and audit.get('admitted') and passed['R1']
                              and not s1['pre_execution_rejections'] and not s1['actions_modified']
                              and s1['maximum_live_shadow_difference'] <= 1e-10))
            rows.append(summary)
            put(r / 'evidence' / (case['id'] + '-paired.json'), summary)
            ledger.update(completed=len(rows), worker_CPU_s=time.process_time() - began)
            write(r / 'RESOURCE_LEDGER.json', ledger)
        import torch
        if torch.cuda.is_initialized():
            raise PermissionError('CR2_CPU_EXPORT_MUST_NOT_INITIALIZE_CUDA')
        put(r / 'evidence/PAIRED_EXPORT_RESULT.private.json', dict(status='TERMINAL_CR2_PAIRED_EXPORT',
            rows=rows, actual_cross_parent_batches=mixed, new_range_fits=0, model_forward=False,
            constructors=0, integrations=0, statistics_in_PV1_unchanged=True))
        return dict(completed=len(rows), admitted=sum(r['admitted'] for r in rows), new_physics=0)
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        put(r / 'evidence/EXPORT_FAILURE.private.json', dict(error=failure, completed=len(rows)))
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = time.process_time() - began + after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_EXPORT', CPU_s=cpu,
                      CPU_reserved_s=0, completed=len(rows), failure=failure)
        write(r / 'RESOURCE_LEDGER.json', ledger); lock.close()
