"""Read both ADAPT5K sites through the production TRAIN readers without a model."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter import adapt5k_data
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
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


def target_examples(target, stage_start):
    if stage_start == 0:
        return [0, target.original_windows-1, target.original_windows, len(target)-1]
    samples = [0, target.original_windows-1]
    begin = target.original_windows
    for member in target.members[1:]:
        for episode in member.manifest['episodes']:
            end = begin + episode['windows'] - 1
            samples.extend((begin, end))
            begin = end+1
    return sorted(set(samples))


def d2_windows(target, spec):
    d2 = spec['d2_audit']
    member = target.members[-1]
    token = d2['tokenizer']
    if (spec['stage_start'] != contract.SWITCH_PHASE or member.manifest_sha != d2['manifest_sha256']
            or sha(token['path']) != token['sha256'] or len(member.manifest['episodes']) != 2
            or len(member) != 139):
        raise ValueError('ADAPT5K_EXPECTS_THE_ADMITTED_TWO_HOLD_D2_BATCH')
    tokenize = Tokenizer(token['path']).delegate.tokenize
    checked = []
    begin = len(target)-len(member)
    for episode in member.manifest['episodes']:
        eid, count = episode['id'], episode['windows']
        data = member.labels[eid]
        references = [target.reference(begin+i) for i in range(count)]
        nominal, consumed = data['nominal_policy_H40'], data['consumed_mask']
        ticks = data['native_step']
        if (episode['task'] not in ('HOLD-red', 'HOLD-blue') or episode['split'] != 'TRAIN'
                or not np.array_equal(ticks, 5703+250*np.arange(count))
                or not np.array_equal(data['supervision_mask'], np.ones((count,40),dtype=bool))
                or not np.array_equal(consumed,np.tile([True]*10+[False]*30,(count,1)))
                or any(abs(ref['query_native_s'] - tick*.002) > 1e-8
                       or not np.isfinite(ref['source_native_s']) for ref,tick in zip(references,ticks,strict=True))):
            raise ValueError(f'ADAPT5K_D2_COMPLETE_PLAN_AND_NATIVE_CLOCK_REQUIRED:{eid}')
        due = [i for i,ref in enumerate(references) if contract.due_close(nominal[i], consumed[i], ref['width_m'])]
        if not due or due[0] == 0:
            raise ValueError(f'ADAPT5K_D2_MISSING_CLOSURE_AND_PRECEDING_WINDOW:{eid}')
        required = {'first':0, 'before_closure':due[0]-1, 'P10_due_closure':due[0], 'last':count-1}
        samples = []
        for i in sorted(set(required.values())):
            obs = target[begin+i]
            tokens = {}
            for name,state in [('state_present',obs['state'][:7]),('state_dropped',None)]:
                ids, mask = tokenize(obs['prompt'],state,use_discrete_state_format=True)
                tokens[name] = dict(ids=bank_sha256(ids),mask=bank_sha256(mask))
            samples.append(dict(identity=target.identity(begin+i), anchors=[k for k,v in required.items() if v==i],
                observations=observation_id(obs), tokens=tokens, previous_actual_command=references[i]))
        checked.append(dict(episode=eid, windows=count, confirmed_P10_atoms=count*10, due_close_anchors=due,
            label_sha256=episode['labels']['sha256'], native_result_sha256=episode['native_result_sha256'],
            clocks_sha256=canonical_hash(references), samples=samples))
        begin += count
    return dict(manifest_sha256=member.manifest_sha, tokenizer_sha256=token['sha256'], episodes=checked,
                verified_semantic_windows=139, confirmed_P10_atoms=1390)


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
    picks = dict(target=target_examples(target,spec['stage_start']),
                 robodojo=source_examples(source),
                 libero=sorted({int(i) for i in banks['libero'][:6]}))
    rows = {d:[dict(index=i, identity=datasets[d].identity(i), inputs=observation_id(datasets[d][i]))
               for i in picks[d]] for d in DOMAINS}
    result = dict(schema='SLA_ADAPT5K_ACTUAL_TRAIN_READERS_V1',
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
    if 'd2_audit' in spec:
        result['d2_audit'] = d2_windows(target,spec)
    return result


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
