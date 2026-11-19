"""Run fixed nonphysical compatibility checks for two A40 engineering states."""
import argparse
import os
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6
from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import roundtrip_error
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, HORIZONS
from go2arx5_vla.sla_adapter.scale20k_data import readers
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha, measure, validate
from go2arx5_vla.sla_adapter.scale20k_training import host_batch
from go2arx5_vla.sla_adapter.site_v2 import validate as validate_site
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.benchmark_scale20k import ARM, SCOPE, START, UPDATES
from scripts.sla.close_goal_g1_v1 import width_metrics
from scripts.sla.evaluate_scale20k import relocated_training, restore_engine


PANEL = 'SLA_A40_MATCHED_FIXED_REPLAY_V1'
FROZEN_CHECKS = dict(xyz_m=.001, so3_rad=.01, gripper_m=.0005, crossing_s=.05)
COUNTS = dict(target=32,robodojo=8,libero=8)
VARIANTS = ('a40_microbatch_1','a40_microbatch_4')


def frozen_source_inputs(spec, resize_with_pad):
    panel = read_pin(spec['original_panel'])
    train = relocated_training(read_pin(spec['parent_training_spec']), read_pin(spec['site_assets']))
    ds = readers(train['data'],resize_with_pad,source_split='SOURCE_DEV')
    validate(panel,train['data'],ds)
    return train,ds,panel


def choose(panel, target):
    step_rows = panel['rows']['target']
    positive = []
    for ordinal,row in enumerate(step_rows):
        _,label,_,_ = target.raw(row['index'])
        if width_metrics(label,label)['expert_crossing_in_window'] is not None:
            positive.append(ordinal)
    ordered = list(dict.fromkeys([*positive, *range(len(step_rows))]))
    selection = dict(target=sorted(ordered[:COUNTS['target']]),robodojo=list(range(COUNTS['robodojo'])),
                     libero=list(range(COUNTS['libero'])))
    return selection,positive


def prepare(spec, resize_with_pad):
    _, ds, panel = frozen_source_inputs(spec,resize_with_pad)
    return selection(spec,panel,ds['target'])


def selection(spec, panel, target):
    chosen, events = choose(panel,target)
    if not events:
        raise ValueError('FIXED_A40_REPLAY_PANEL_REQUIRES_AT_LEAST_ONE_ACTUAL_LABEL_CROSSING')
    return dict(schema=PANEL,plan=PLAN,original_panel=spec['original_panel'],site_assets_sha256=spec['site_assets']['sha256'],
        ordinals=chosen,label_crossing_ordinals=events,prescribed_checks=FROZEN_CHECKS,
        variants=list(VARIANTS),predictions_per_role=sum(COUNTS.values()),native_requests=0,optimizer_updates=0)


def evaluate(spec, mode):
    import flax.nnx as nnx
    import jax
    from openpi.models import pi0
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    root = Path(spec['registered_root']).resolve(strict=True)
    pin = spec['selection']
    chosen = read_pin(pin)
    benchmark = read_pin(spec['benchmark_terminal'])
    variant = benchmark['variant']
    model = benchmark['engineering_checkpoint']
    marker = read_pin(dict(path=str(Path(model['path'])/'COMPLETE.json'),sha256=model['sha256']))
    binding = marker['binding']
    if (spec['plan'] != PLAN or spec['hardware_allowed'] is not False or spec['physics'] is not False
            or root.stat().st_uid != os.getuid() or any(root.iterdir()) or variant not in VARIANTS
            or benchmark['scope'] != SCOPE or benchmark['actual_optimizer_updates'] != UPDATES or benchmark['exit_code']
            or not benchmark['engineering_only'] or benchmark['eligible_as_formal_parent']
            or marker['step'] != START+UPDATES or not binding.get('engineering_only') or binding['variant'] != variant
            or chosen['predictions_per_role'] != 48 or spec['model_generation_cap'] != 48):
        raise PermissionError('EXACT_FROZEN_ENGINEERING_A40_CHECKPOINT_AND_48_REPLAY_CAP_REQUIRED')
    stop = False

    def stop_signal(*_):
        nonlocal stop
        stop = True

    def guard():
        if stop or Path(spec['STOP_path']).exists() or time.time() >= spec['deadline_epoch']-60:
            raise TimeoutError('BOUNDED_MATCHED_REPLAY_STOP')

    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    rows = {d:[] for d in DOMAINS}
    actual = dict(mode=mode,variant=variant,engineering_checkpoint=model,selection=pin,rows=rows,
        attempted_generations=0,model_generations=0,matched_restore_outputs=0,new_optimizer_updates=0,
        native_requests=0,formal_endpoint=False,old_TEST_access=False)
    lock = OwnerLock(spec['neural_owner_lock'])
    try:
        guard()
        actual['site'] = validate_site(spec)
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise PermissionError('ACTUAL_SINGLE_A40_FOR_ENGINEERING_REPLAY_REQUIRED')
        train,ds,panel = frozen_source_inputs(spec,resize_with_pad)
        if (chosen != selection(spec,panel,ds['target']) or train['arm'] != ARM
                or {d:train['data'][d]['manifest']['sha256'] for d in DOMAINS} != binding['data_manifest_sha256']
                or sha(train['tokenizer']['path']) != binding['tokenizer_sha256']):
            raise ValueError('ENGINEERING_REPLAY_MUST_USE_ORIGINAL_SOURCE_AND_FROZEN_SELECTED_ROWS')
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        graphs,parameters,frozen = restore_engine(train,binding,model)
        tokenizer = Tokenizer(train['tokenizer']['path'])
        fields,_ = extracted_fields(batch_v2,pi0)
        original = None
        if mode == 'restore':
            original = read_pin(spec['ordinary_result'])
            if (original['mode'] != 'ordinary' or original['exit_code'] or original['engineering_checkpoint'] != model
                    or original['selection'] != pin or original['model_generations'] != 48):
                raise ValueError('INDEPENDENT_ENGINEERING_RESTORE_NEEDS_THE_ORIGINAL_48_OUTPUTS')
        for domain in DOMAINS:
            sampler = full_h40_v6.make_sampler(graphs[domain],fields,rematerialize=False,precision='highest')
            args = nnx.State.merge(parameters['shared'],parameters[domain])
            directory = root/domain
            directory.mkdir()
            for index,ordinal in enumerate(chosen['ordinals'][domain]):
                guard()
                sample = panel['rows'][domain][ordinal]
                obs,label = host_batch(ds[domain],tokenizer,domain,(ordinal,),(sample['index'],),training=False)
                shape = (1,HORIZONS[domain],32)
                noise = jax.random.normal(jax.random.PRNGKey(sample['noise_id']),shape)
                actual['attempted_generations'] += 1
                y = np.asarray(sampler(args,frozen,obs,noise))
                if y.shape != shape or not np.isfinite(y).all():
                    raise ValueError('ENGINEERING_REPLAY_REQUIRES_FULL_FINITE_NATIVE_HORIZON')
                actual['model_generations'] += 1
                physical,metrics = measure(domain,ds[domain],sample,y[0],np.asarray(label[0]),panel)
                if not np.isfinite(physical).all():
                    raise ValueError('ENGINEERING_REPLAY_NONFINITE_DECODED_ACTION')
                output = directory/f'{ordinal:03d}.npz'
                with output.open('xb') as fd:
                    np.savez(fd,normalized=y[0],decoded=physical)
                    fd.flush()
                    os.fsync(fd.fileno())
                hashes = dict(normalized=bytes_sha(y[0]),decoded=bytes_sha(physical))
                rows[domain].append(dict(ordinal=ordinal,identity=sample['identity'],noise_id=sample['noise_id'],
                    artifact=dict(path=str(output),sha256=sha(output)),hashes=hashes,metrics=metrics))
                if original is not None:
                    ref = original['rows'][domain][index]
                    if ref['ordinal'] != ordinal or ref['hashes'] != hashes or sha(ref['artifact']['path']) != ref['artifact']['sha256']:
                        raise ValueError(f'INDEPENDENT_ENGINEERING_RESTORE_CHANGED:{domain}:{ordinal}')
                    actual['matched_restore_outputs'] += 1
        actual.update(status='SEALED_MATCHED_ENGINEERING_OUTPUTS',exit_code=0)
    except (Exception,KeyboardInterrupt) as error:
        actual.update(status='ENGINEERING_OUTPUT_TECHNICAL_TERMINAL',exit_code=1,
                      failure=f'{type(error).__name__}: {error}')
    finally:
        write(root/'RESULT.private.json',actual)
        lock.close()
    return actual['exit_code']


def read_arrays(row):
    pin = row['artifact']
    if sha(pin['path']) != pin['sha256']:
        raise ValueError('ENGINEERING_REPLAY_OUTPUT_FILE_CHANGED')
    with np.load(pin['path'],allow_pickle=False) as data:
        return data['normalized'],data['decoded']


def compare(first, second, first_restore, second_restore):
    all_roles = first,second,first_restore,second_restore
    if (tuple(x['variant'] for x in (first, second)) != VARIANTS
            or any(x['exit_code'] or x['model_generations'] != 48 or x['new_optimizer_updates'] or x['native_requests']
                   or x['selection'] != first['selection'] for x in all_roles)
            or any(t['mode'] != 'restore' or t['matched_restore_outputs'] != 48 or
                   t['engineering_checkpoint'] != o['engineering_checkpoint']
                   for o,t in ((first,first_restore),(second,second_restore)))):
        raise ValueError('ENGINEERING_COMPATIBILITY_REQUIRES_COMPLETE_BASELINE_CANDIDATE_AND_RESTORES')
    reports = {}
    for domain in DOMAINS:
        rows = []
        for a,b in zip(first['rows'][domain],second['rows'][domain],strict=True):
            if any(a[k] != b[k] for k in ('ordinal','identity','noise_id')):
                raise ValueError('MATCHED_ENGINEERING_EXAMPLE_OR_NOISE_CHANGED')
            n1,d1 = read_arrays(a)
            n2,d2 = read_arrays(b)
            row = dict(ordinal=a['ordinal'],normalized_bit_exact=a['hashes']['normalized']==b['hashes']['normalized'],
                decoded_bit_exact=a['hashes']['decoded']==b['hashes']['decoded'],
                max_normalized_difference=float(np.abs(n1-n2).max()))
            if domain == 'target':
                delta = roundtrip_error(d1,d2)
                m1,m2 = a['metrics'],b['metrics']
                first_cross,second_cross = (m['width']['predicted_first_crossing'] for m in (m1,m2))
                presence = (first_cross is None) == (second_cross is None)
                crossing = .05*abs(first_cross-second_cross) if first_cross is not None and second_cross is not None else None
                h40 = m1['envelope']['original_full_H40_pass'] == m2['envelope']['original_full_H40_pass']
                seam = m1['gripper_seam']['original_execution_speed_exceeded'] == m2['gripper_seam']['original_execution_speed_exceeded']
                row.update(xyz_m=delta['xyz_m'],so3_rad=delta['so3_rad'],gripper_m=.040*delta['gripper'],
                            crossing_s=crossing,predicted_cross_presence_equal=presence,H40_equal=h40,seam_equal=seam)
                row['fixed_physical_criteria_pass'] = (h40 and seam and presence
                    and all(row[k] is None or row[k] <= limit for k,limit in FROZEN_CHECKS.items()))
            rows.append(row)
        if len(rows) != COUNTS[domain]:
            raise ValueError('ENGINEERING_OUTPUT_REPLAY_DENOMINATOR_CHANGED')
        reports[domain] = rows
    return dict(schema='SLA_A40_MATCHED_NUMERICAL_OUTPUT_ACTUAL_V1',selection=first['selection'],
        checkpoints=[x['engineering_checkpoint'] for x in (first,second)],
        model_generations=192,native_requests=0,new_optimizer_updates=0,matched_restore_outputs=96,
        numerical_output_gate_all=all(r['fixed_physical_criteria_pass'] for r in reports['target']),
        original_output_thresholds=FROZEN_CHECKS,by_domain=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','ordinary','restore','compare'))
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    spec = read_pin(dict(path=args.spec,sha256=args.sha256))
    if args.mode in ('ordinary','restore'):
        raise SystemExit(evaluate(spec,args.mode))
    if args.output is None or args.output.exists():
        raise FileExistsError('FRESH_ENGINEERING_REPORT_PATH_REQUIRED')
    if args.mode == 'prepare':
        from openpi_client.image_tools import resize_with_pad

        result = prepare(spec,resize_with_pad)
    else:
        result = compare(*(read_pin(spec[name]) for name in ('base','candidate','base_restore','candidate_restore')))
    write(args.output,result)


if __name__ == '__main__':
    main()
