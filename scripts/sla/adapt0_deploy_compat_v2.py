"""Measure a new deployment gate on the original 16 RGB conditions."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import roundtrip_error
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_evaluation import RESTORE, bytes_sha
from scripts.pi0292.durable import write
from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from scripts.sla.replay_scale20k_core import guard_signature


SCHEMA = 'SLA_LONGRUN_DEPLOY_COMPAT_V2'
LIMITS = dict(xyz_m=.001, so3_rad=.01, gripper_m=.0005, closing_s=.05)
ROLES = ('ordinary','restore')


def prediction_deltas(reference, actual, previous_width):
    score = roundtrip_error(reference,actual)

    def closings(track):
        widths = np.r_[previous_width,.002+.040*track[:,9]]
        return np.flatnonzero((widths[:-1]>=.022) & (widths[1:]<.022))

    a,b = closings(reference),closings(actual)
    first = (int(a[0]) if a.size else None, int(b[0]) if b.size else None)
    time = .05*abs(first[0]-first[1]) if a.size and b.size else None
    ra,rb = (matrix_from10(x)[:,:3,:3] for x in (reference,actual))
    rotation = Rotation.from_matrix(ra @ np.swapaxes(rb,-1,-2)).magnitude()
    diffs = dict(xyz_m=np.max(abs(actual[:,:3]-reference[:,:3]),axis=-1),
                 so3_rad=rotation,gripper_m=.04*abs(actual[:,9]-reference[:,9]))
    report = {k:dict(max=float(v.max()),p95=float(np.quantile(v,.95)),rms=float(np.sqrt(np.mean(v**2))))
              for k,v in diffs.items()}
    values=dict(xyz_m=score['xyz_m'],so3_rad=score['so3_rad'],gripper_m=.040*score['gripper'],closing_s=time)
    return dict(maxima=values,trajectory_errors=report,closing_atoms=first,
                closing_presence_equal=bool(a.size)==bool(b.size),P10_closing_equal=bool(np.any(a<10))==bool(np.any(b<10)),
                inside_limits=all(values[k] is None or values[k]<=v for k,v in LIMITS.items()))


def arrays(row):
    path = row['output']['path']
    if sha(path) != row['output']['sha256']:
        raise ValueError('DEPLOY_COMPAT_V2_OUTPUT_FILE_CHANGED')
    with np.load(path,allow_pickle=False) as data:
        normalized,physical = data['normalized'],data['decoded']
    if (normalized.shape!=(40,32) or physical.shape!=(40,10) or not np.isfinite(normalized).all()
            or not np.isfinite(physical).all() or bytes_sha(normalized)!=row['normalized_sha256']
            or bytes_sha(physical)!=row['decoded_sha256']):
        raise ValueError('DEPLOY_COMPAT_V2_REQUIRES_ORIGINAL_FINITE_ARRAYS_AND_HASHES')
    return normalized,physical


def evaluate(spec):
    declaration, formal, ordinary, restore = map(read_pin,(spec[k] for k in ('declaration','A40_formal','ordinary','restore')))
    old = read_pin(declaration['strict_failure'])
    input_audit = read_pin(declaration['input_audit'])
    archived = read_pin(declaration['archive_index'])
    if (declaration['schema']!=SCHEMA or declaration['limits']!=LIMITS or declaration['generation_cap']!=32
            or declaration['ordinary_A40_formal_sha256']!=spec['A40_formal']['sha256']
            or old['status']!='CORE_ADAPT0_RGB16_OUTPUT_GATE_FAILED' or old['model_generations']!=16
            or declaration['prior_qualification_replays']!=16 or old['native_requests']
            or declaration['spent_generated_attempts']+32>336
            or tuple(x['phase'] for x in (formal,ordinary,restore))!=(0,0,0)
            or any(x['exit_code'] or x['generation_attempts']!=16 or x['model_generations']!=16
                   or x['optimizer_updates'] or x['native_requests'] or len(x['rows'])!=16
                   or x['deployment_compat_v2']!=spec['declaration'] for x in (ordinary,restore))
            or ordinary['environment']!=restore['environment'] or not ordinary['environment']
            or ordinary['process_identity']==restore['process_identity']
            or tuple(x['compat_role'] for x in (ordinary,restore))!=ROLES
            or ordinary['package']!=restore['package'] or formal['panel']['sha256']!=declaration['panel_sha256']
            or formal['checkpoint']['sha256']!=declaration['checkpoint_sha256']
            or input_audit['formal_sha256']!=spec['A40_formal']['sha256']
            or input_audit['panel_sha256']!=declaration['panel_sha256']
            or input_audit['package_sha256']!=ordinary['package']
            or input_audit['recorded_predictions']!=len(RESTORE)
            or set(archived)!=set(map(str,RESTORE))):
        raise ValueError('DEPLOY_COMPAT_V2_REQUIRES_PREDECLARATION_AND_TWO_ISOLATED_CORE_PROCESSES')
    rows = []
    for i,one,two,prior_inputs in zip(RESTORE,ordinary['rows'],restore['rows'],input_audit['rows'],strict=True):
        before = formal['rows']['target'][i]
        portable = archived[str(i)]
        if (any(x['ordinal']!=i or x['identity']!=before['identity'] or x['noise_id']!=before['noise_id']
                for x in (one,two)) or one['previous_actual_command']!=two['previous_actual_command']
                or one['input_sha256']!=two['input_sha256'] or one['noise_sha256']!=two['noise_sha256']
                or portable['sha256']!=before['output']['sha256']
                or prior_inputs['ordinal']!=i or not prior_inputs['offline_provider_input_exact']
                or one['input_sha256']!={k:v['provider'] for k,v in prior_inputs['input_fingerprints'].items()}):
            raise ValueError(f'DEPLOY_COMPAT_V2_INPUT_OR_CLOCK_CHANGED:{i}')
        a,p = arrays(dict(before,output=portable))
        b,q = arrays(one)
        c,r = arrays(two)
        delta = prediction_deltas(p,q,one['previous_actual_command']['width_m'])
        signature = guard_signature(before['metrics']['envelope'],before['metrics']['gripper_seam'])
        expected = list(signature); expected[1] = list(expected[1]) if expected[1] is not None else None
        guards = one['guard_signature']==two['guard_signature']==expected
        same = (one['normalized_sha256']==two['normalized_sha256'] and one['decoded_sha256']==two['decoded_sha256']
                and np.array_equal(b,c) and np.array_equal(q,r))
        legal = guards and same and delta['inside_limits'] and delta['closing_presence_equal'] and delta['P10_closing_equal']
        rows.append(dict(ordinal=i,cross_GPU_guard_exact=guards,same_Core_restore_bitwise=same,passed=legal,
                         normalized_max_difference=float(abs(a-b).max()),**delta))
    return dict(schema=SCHEMA,passed=all(r['passed'] for r in rows),rows=rows,denominator=len(rows),
                model_generations=32,native_requests=0,optimizer_updates=0,proof=spec,
                strict_failure_preserved=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output,evaluate(read_pin(dict(path=args.spec,sha256=args.sha256))))
