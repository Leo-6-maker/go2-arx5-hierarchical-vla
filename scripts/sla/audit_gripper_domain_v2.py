"""Read all 48 fixed saved TRAIN predictions, without model or physics calls."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import data, read_pin
from go2arx5_vla.sla_adapter.gripper_v2 import range_audit
from go2arx5_vla.world_model.task_learning_endpoint_basis_v3 import decode


def summarize(records):
    return dict(predictions=len(records), knots=40 * len(records),
                raw_full_H40_gripper_legal=sum(r['raw']['full_H40_legal'] for r in records),
                last_gripper_legal=sum(r['raw']['last_legal'] for r in records),
                endpoints_gripper_legal=sum(r['raw']['endpoints_legal'] for r in records),
                interior_gripper_legal=sum(r['raw']['interior_legal'] for r in records),
                last_legal_but_full_raw_rejected=sum(r['raw']['last_legal'] and not r['raw']['full_H40_legal'] for r in records),
                first_outside=sum(r['raw']['first_outside'] for r in records),
                last_outside=sum(r['raw']['last_outside'] for r in records),
                interior_outside=sum(r['raw']['interior_outside'] for r in records),
                below_zero=sum(r['raw']['below_zero'] for r in records),
                above_one=sum(r['raw']['above_one'] for r in records),
                maximum_lower_violation=max(r['raw']['maximum_lower_violation'] for r in records),
                maximum_upper_violation=max(r['raw']['maximum_upper_violation'] for r in records),
                basis_constructed_offline=sum(r['basis_status'] == 'CONSTRUCTED_OFFLINE_DOMAIN_ONLY' for r in records),
                basis_gripper_legal=sum(r.get('basis_gripper', {}).get('full_H40_legal', False) for r in records),
                basis_refused_by_original_raw_gate=sum(r['basis_status'] == 'REFUSED_BY_ORIGINAL_FULL_RAW_GATE' for r in records),
                missing_previous_nominal_witness=sum(r['basis_status'] == 'PREVIOUS_NOMINAL_WITNESS_UNAVAILABLE' for r in records))


def run(spec):
    prior = read_pin(spec['saved_probe_audit'])
    candidate = read_pin(spec['probe_candidate'])
    root = Path(spec['probe_candidate']['path']).parent
    lease = json.loads((root / 'control/LEASE.json').read_bytes())
    if (prior['schema'] != 'SLA_FIXED_SAVED_TRAIN_PROBES_CPU_AUDIT_V1'
            or prior['saved_predictions_read'] != 48 or not prior['passes']
            or prior['candidate'] != spec['probe_candidate']
            or prior['target_audit'] != spec['target_audit']
            or candidate['step'] != 500 or candidate['cursor'] != 16000
            or lease['status'] != 'CLOSED' or lease['worker_exit_code'] != 0):
        raise ValueError('REUSE_EXACT_COLD_FIXED_S1_PROBE_AUDIT')
    dataset = data(spec)
    contexts = {}
    for index in prior['fixed_indices']:
        state, _, _, _ = dataset.original.raw(index)
        episode, anchor = dataset.original.rows[index]
        packet = dataset.original._cached[1]
        atom = int(packet['native_command_anchor'][anchor])
        # Never substitute measured aperture or the next expert command.
        previous = float(packet['committed_absolute10'][atom - 1, 9]) if atom else None
        if previous is not None and not 0 <= previous <= 1:
            raise ValueError('PREVIOUS_CONSUMED_NOMINAL_COMMAND_RANGE')
        contexts[index] = dict(state=state, previous=previous,
                               phase=episode['expert_stage_audit_only'][anchor])
    records = []
    expected = [(index, noise) for index in prior['fixed_indices'] for noise in (0, 1)]
    for step in (0, 250, 500):
        folder = root / 'work' / f'step{step}-RGB'
        manifest = json.loads((folder / 'MANIFEST.private.json').read_bytes())
        if [(r['index'], r['noise']) for r in manifest] != expected:
            raise ValueError('FULL_FIXED_PROBE_POPULATION_REQUIRED')
        for row in manifest:
            with np.load(folder / row['file'], allow_pickle=False) as packet:
                raw, normalized = packet['physical_H40'], packet['normalized']
                if (raw.shape != (40, 10) or normalized.shape != (1, 40, 32)
                        or not np.isfinite(raw).all() or not np.isfinite(normalized).all()
                        or hashlib.sha256(raw.tobytes()).hexdigest() != row['physical_sha256']
                        or hashlib.sha256(normalized.tobytes()).hexdigest() != row['normalized_sha256']):
                    raise ValueError('SAVED_PREDICTION_IDENTITY_CHANGED')
            context = contexts[row['index']]
            record = dict(step=step, index=row['index'], noise=row['noise'], phase=context['phase'],
                          physical_sha256=row['physical_sha256'], raw=range_audit(raw[:, 9]))
            if not record['raw']['full_H40_legal']:
                try:
                    decode(raw, context['state'], consumed_nominal_grip=0.)
                except ValueError as exc:
                    record.update(basis_status='REFUSED_BY_ORIGINAL_FULL_RAW_GATE', original_refusal=str(exc))
                else:
                    raise AssertionError('ORIGINAL_RAW_GATE_MUST_NOT_BE_BYPASSED')
            elif context['previous'] is None:
                record['basis_status'] = 'PREVIOUS_NOMINAL_WITNESS_UNAVAILABLE'
            else:
                basis, _ = decode(raw, context['state'], consumed_nominal_grip=context['previous'])
                record.update(basis_status='CONSTRUCTED_OFFLINE_DOMAIN_ONLY',
                              basis_gripper=range_audit(basis[:, 9]),
                              previous_nominal_source='PREVIOUS_CONSUMPTION_PROVEN_EXPERT_COMMAND')
            records.append(record)
    return dict(schema='SLA_SAVED_GRIPPER_DOMAIN_DIAGNOSTIC_V2',
                plan_revision=spec['scaleup_plan_revision'], candidate=spec['probe_candidate'],
                reused_probe_audit=spec['saved_probe_audit'], saved_predictions_read=48,
                unique_TRAIN_windows_read=8, optimizer_updates=0, training_optimization_visits=0,
                model_loads=0, model_calls=0, teacher_calls=0, physical_requests=0, native_integrations=0,
                DEV_TEST_reads=0, checkpoint_selection=False, raw_prediction_changes=0,
                action_clips=0, native_reference_certificates=0, native_task_success_established=False,
                legality_scope='GRIPPER_RANGE_AND_ORIGINAL_BASIS_RAW_GATE_ONLY_NOT_IK_COLLISION_OR_WBC_CERTIFICATION',
                summary={str(step): summarize([r for r in records if r['step'] == step]) for step in (0, 250, 500)},
                phase_summary={str(step): {phase: summarize([r for r in records if r['step'] == step and r['phase'] == phase])
                                         for phase in sorted({r['phase'] for r in records})}
                               for step in (0, 250, 500)}, records=records)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root']) / 'RESULT.private.json', run(spec))
