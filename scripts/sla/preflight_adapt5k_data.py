"""Read both ADAPT5K sites through the production TRAIN readers without a model."""
import argparse
from collections import Counter
from pathlib import Path

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter import adapt5k_data
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, bank_sha256, canonical_hash
from scripts.pi0292.durable import write


def observation_id(item):
    return dict(rgb={k:bank_sha256(a) for k,a in item['image'].items()},
                image_mask=item['image_mask'],
                padding={k:bank_sha256(a) for k,a in item['image_padding_mask'].items()},
                state=bank_sha256(item['state']), actions=bank_sha256(item['actions']), prompt=item['prompt'])


def source_examples(source):
    starts, ends = {}, {}
    begin = 0
    for row in source.episodes.values():
        if row['id'].startswith('ADAPT-RD-'):
            starts.setdefault(row['task'], begin)
            ends[row['task']] = begin + row['windows'] - 1
        begin += row['windows']
    if set(starts) != {r['task'] for r in source.episodes.values()}:
        raise ValueError('EVERY_R200_TASK_MUST_HAVE_NEW_REAL_SOURCE_INPUTS')
    return [i for task in sorted(starts) for i in (starts[task], ends[task])]


def preflight(spec, resize_with_pad):
    stages = ((0, contract.SWITCH_PHASE), (contract.SWITCH_PHASE, contract.UPDATES))
    if (spec['plan'] != contract.PLAN or (spec['stage_start'], spec['stage_stop']) not in stages
            or type(spec['target_seed']) is not int):
        raise ValueError('ADAPT5K_BOUNDED_STAGE_AND_FIXED_SEED_REQUIRED')
    if spec['parent_step'] not in contract.FORMAL_PARENTS:
        raise ValueError('ADAPT5K_FIRST_STAGE_REQUIRES_A_FORMAL_PARENT')
    datasets = adapt5k_data.readers(spec['data'], resize_with_pad)
    target, source, libero = (datasets[d] for d in DOMAINS)
    banks, digests, bank_report = adapt5k_data.banks_for(datasets, stage_start=spec['stage_start'],
        stage_stop=spec['stage_stop'], target_seed=spec['target_seed'])
    picks = dict(target=[0,target.original_windows-1,target.original_windows,len(target)-1],
                 robodojo=source_examples(source),
                 libero=sorted({int(i) for i in banks['libero'][:6]}))
    rows = {d:[dict(index=i, identity=datasets[d].identity(i), inputs=observation_id(datasets[d][i]))
               for i in picks[d]] for d in DOMAINS}
    return dict(schema='SLA_ADAPT5K_ACTUAL_TRAIN_READERS_V1',
        parent_step=spec['parent_step'], parent_checkpoint_sha256=spec['parent_checkpoint_sha256'],
        stage_start=spec['stage_start'], stage_stop=spec['stage_stop'],
        target_state_sha256=spec['data']['target']['statistics']['sha256'],
        inherited_R48_statistics_sha256=source.statistics_sha,
        diagnostic_R200_statistics_sha256=source.manifest['statistics']['sha256'],
        total_windows={d:len(datasets[d]) for d in DOMAINS},
        independent_episodes=dict(target=len(target.manifest['episodes']),robodojo=len(source.episodes),
                                  libero=len(libero.episodes)),
        new_target_episodes=target.new_episodes,
        source_tasks=dict(Counter(row['task'] for row in source.episodes.values())),
        bank_digests=digests, target_bank_report=bank_report,
        banks_windows={d:len(banks[d]) for d in DOMAINS},
        input_sha256=canonical_hash(rows), inputs=rows, optimizer_updates=0, optimization_visits=0,
        model_generations=0, native_requests=0, TEST_access=False)


def main():
    from openpi_client.image_tools import resize_with_pad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    out = preflight(read_pin(dict(path=a.spec,sha256=a.sha256)),resize_with_pad)
    write(a.output, out)
    print(out['input_sha256'])


if __name__ == '__main__':
    main()
