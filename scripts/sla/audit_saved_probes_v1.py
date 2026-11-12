"""Audit all already-generated fixed TRAIN probes; never load a policy."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import data, probe_indices, read_pin
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def run(spec):
    candidate = read_pin(spec['probe_candidate'])
    root = Path(spec['probe_candidate']['path']).parent
    lease = json.loads((root/'control/LEASE.json').read_bytes())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    if (lease['status'] != 'CLOSED' or lease['worker_exit_code'] != 0 or worker['exit_code'] != 0
            or candidate['operation'] != 'CANDIDATE' or candidate['step'] != 500
            or candidate['cursor'] != 16000):
        raise ValueError('ACTUAL_FIXED500_COLD_S1_PROBES_REQUIRED')
    complete = json.loads((Path(candidate['last_checkpoint']['path'])/'COMPLETE.json').read_bytes())
    if (sha(Path(candidate['last_checkpoint']['path'])/'COMPLETE.json') != candidate['last_checkpoint']['sha256']
            or complete['bindings']['target_audit'] != spec['target_audit']['sha256']
            or lease['spec_sha256'] != sha(root/'EXECUTION_SPEC.json')):
        raise ValueError('ACTUAL_SAVED_CHECKPOINT_AND_TARGET_IDENTITY_REQUIRED')
    dataset = data(spec)
    indices = probe_indices(dataset)
    targets, stages = {}, {}
    for index in indices:
        _, target, _, _ = dataset.original.raw(index)
        targets[index] = target
        episode, anchor = dataset.original.rows[index]
        stages[index] = episode['expert_stage_audit_only'][anchor]
    expected = [(i,n) for i in indices for n in range(2)]
    records, summaries = [], {}
    for step in (0,250,500):
        folder = root/'work'/f'step{step}-RGB'
        manifest = json.loads((folder/'MANIFEST.private.json').read_bytes())
        if [(r['index'],r['noise']) for r in manifest] != expected:
            raise ValueError('EVERY_FIXED_PROBE_AND_NOISE_AT_EVERY_ENDPOINT_REQUIRED')
        group = []
        for row in manifest:
            with np.load(folder/row['file'], allow_pickle=False) as values:
                predicted, normalized = values['physical_H40'], values['normalized']
                if (predicted.shape != (40,10) or normalized.shape != (1,40,32)
                        or hashlib.sha256(predicted.tobytes()).hexdigest() != row['physical_sha256']
                        or hashlib.sha256(normalized.tobytes()).hexdigest() != row['normalized_sha256']
                        or not np.isfinite(predicted).all() or not np.isfinite(normalized).all()):
                    raise ValueError('ORIGINAL_SAVED_PROBE_ARRAYS_CHANGED')
            target = targets[row['index']]
            xyz = np.linalg.norm(predicted[:,:3]-target[:,:3], axis=1)
            a, b = matrix_from10(predicted), matrix_from10(target)
            angle = Rotation.from_matrix(a[:,:3,:3]@np.swapaxes(b[:,:3,:3],-1,-2)).magnitude()
            grip = predicted[:,9]
            record = dict(step=step,index=row['index'],noise=row['noise'],phase=stages[row['index']],
                xyz_RMSE_m=float(np.sqrt(np.mean(xyz**2))),xyz_max_m=float(xyz.max()),
                SO3_mean_rad=float(angle.mean()),SO3_max_rad=float(angle.max()),
                gripper_MAE=float(np.abs(grip-target[:,9]).mean()),
                raw_gripper_min=float(grip.min()),raw_gripper_max=float(grip.max()),
                raw_gripper_outside_original_range=int(((grip<0)|(grip>1)).sum()),
                fullH40_gripper_range_valid=bool(((grip>=0)&(grip<=1)).all()),
                last_knot_xyz_error_m=float(xyz[-1]),last_knot_gripper=float(grip[-1]))
            records.append(record)
            group.append(record)
        summaries[str(step)] = dict(predictions=len(group),
            mean_xyz_RMSE_m=float(np.mean([r['xyz_RMSE_m'] for r in group])),
            mean_SO3_rad=float(np.mean([r['SO3_mean_rad'] for r in group])),
            mean_gripper_MAE=float(np.mean([r['gripper_MAE'] for r in group])),
            fullH40_gripper_range_valid=sum(r['fullH40_gripper_range_valid'] for r in group),
            raw_gripper_outside_original_range=sum(r['raw_gripper_outside_original_range'] for r in group))
    return dict(schema='SLA_FIXED_SAVED_TRAIN_PROBES_CPU_AUDIT_V1',passes=True,
        candidate=spec['probe_candidate'],candidate_checkpoint=candidate['last_checkpoint'],
        target_audit=spec['target_audit'],fixed_indices=indices,noise_ids=[0,1],steps=[0,250,500],
        unique_TRAIN_windows_read=8,saved_predictions_read=48,model_loads=0,model_calls=0,
        diagnostic_TRAIN_window_reads=8,training_optimization_visits=0,optimizer_updates=0,
        teacher_calls=0,physical_requests=0,summary=summaries,records=records,
        primary_endpoint_remains500=True,checkpoint_selection=False,DEV_TEST_reads=0,
        native_task_success_established=False,clipping=False,endpoint_basis_applied=False)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    value = run(spec)
    write(Path(spec['registered_root'])/'RESULT.private.json',value)
