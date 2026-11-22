"""Freeze six original DEV conditions spanning an unseen F1 layout and F2."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path

from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned


CELLS = (
    ('CHV1-DEV-F1-G01', 'HOLD-red'),
    ('CHV1-DEV-F1-G01', 'PLACE-blue-left'),
    ('CHV1-DEV-F1-G01', 'PLACE-red-right'),
    ('CHV1-DEV-F2-G00', 'HOLD-blue'),
    ('CHV1-DEV-F2-G00', 'PLACE-red-left'),
    ('CHV1-DEV-F2-G00', 'PLACE-blue-right'),
)


def freeze(path):
    from scripts.sla.core_hopper_student_endpoint import cases

    population = dict(path=str(path),sha256=POP)
    pop = pinned(population)
    candidates = cases(pop,'MAIN-phase03800','DEV')
    rows = [next(row for row in candidates if (row['geometry'],row['instruction'])==cell) for cell in CELLS]
    selected = {row['geometry'] for row in rows}
    held_out = {row['scene']['sha256'] for row in rows}
    outside_dev = {row['cluster'] for split,group in pop['conditions'].items() if split!='DEV' for row in group}
    outside_scenes = {row['scene']['sha256'] for row in pop['clusters'] if row['id'] in outside_dev}
    if (len(selected)!=2 or selected & outside_dev or held_out & outside_scenes
            or held_out & {row['scene']['sha256'] for row in candidates[:6]}
            or len({row['condition'] for row in rows})!=6):
        raise ValueError('The cross-layout subset must be disjoint from first-six and non-DEV scenes')
    return dict(schema='MAIN3800_CROSS_LAYOUT_ROSTER_V1',plan=PLAN,population=population,
                candidates=[{k:v for k,v in row.items() if k not in ('id','arm')} for row in rows],
                tick_cap_per_arm=sum(row['tick_cap'] for row in rows),
                query_cap_per_arm=sum(row['query_cap'] for row in rows),
                request_cap_per_arm=sum(row['request_cap'] for row in rows),
                selected_from_outcomes=False,physical_requests=0,generations=0,updates=0)


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('population',type=Path)
    cli.add_argument('output',type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    roster = freeze(args.population)
    write_once(args.output,roster)
    print(json.dumps(dict(sha256=digest(args.output),conditions=len(roster['candidates']),
                          tick_cap_per_arm=roster['tick_cap_per_arm'],query_cap_per_arm=roster['query_cap_per_arm'])))
