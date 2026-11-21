"""Compare terminal contact pairs behind the original producer's consecutive TRAIN stops."""
from argparse import ArgumentParser
from pathlib import Path

import mujoco

from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.core_hopper_carry_audit import BASE, history, read
from scripts.sla.hopper_parent_custody import write_once


def inspect(root, limit):
    spec = read(root/'EXECUTION_SPEC.json')
    if spec['source_sha'] != BASE or not 1 <= limit <= 240:
        raise ValueError('Expected original TRAIN producer and fixed terminal count')
    records = [r for r in read(root/'RECORDS.private.json') if r['split']=='TRAIN'][:limit]
    if len(records) != limit or len({r['condition'] for r in records}) != limit:
        raise ValueError('Incomplete TRAIN prefix')
    previous, pairs = {}, {}
    with forbid_integration(mujoco) as calls:
        for row in records:
            cell = row['condition'].rsplit('-',1)[0]
            first = previous.get(cell)
            previous[cell] = row
            if (first is None or row['first_failure']!='ILLEGAL_CONTACT'
                    or first['first_failure']!=row['first_failure']):
                continue
            if cell in pairs:
                raise ValueError('Unexpected additional request in a paused cell')
            evidence = []
            for member in (first,row):
                case = history(root,member)
                parts = {tuple(sorted(zip(hit['bodies'],hit['geoms'],strict=True))) for hit in case['pairs']}
                if not parts:
                    raise ValueError('No identified stop contact for '+case['condition'])
                evidence.append(dict(condition=case['condition'], stop_tick=case['terminal_tick'],
                    last_query_tick=case['last_query_tick'], last_query_phase=case['phase'],
                    grasp_at_last_query=case['last_query_native_grasp'], body_geom_pairs=sorted(parts),
                    result_sha256=case['result_sha256'], queries_sha256=case['query_sha256'],
                    post_STOP_ticks=case['post_STOP_ticks']))
            a,b = evidence
            pairs[cell] = dict(cases=evidence, same_last_query_phase=a['last_query_phase']==b['last_query_phase'],
                               same_terminal_parts=a['body_geom_pairs']==b['body_geom_pairs'])
        if calls:
            raise RuntimeError('Unexpected integration during contact attribution')
    return dict(source=BASE, train_closed=limit, paired_cells=len(pairs), cells=pairs, VAL_inputs=0,
                matching_both=sum(v['same_last_query_phase'] and v['same_terminal_parts'] for v in pairs.values()),
                native_requests=0, native_ticks=0, optimizer_updates=0, model_generations=0)


if __name__ == '__main__':
    cli = ArgumentParser()
    cli.add_argument('original', type=Path)
    cli.add_argument('train_prefix', type=int)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    write_once(args.output,inspect(args.original,args.train_prefix))
