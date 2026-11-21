"""Check the registered HOLD approach pitch on four saved F2 red PLACE descents."""
import argparse
from pathlib import Path
import time

import mujoco

from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_carry_audit import BASE, read, read_pin
from scripts.sla.core_hopper_red_approach import PITCH_DEGREES, world_pitch
from scripts.sla.core_hopper_red_approach_audit import analyze_case
from scripts.sla.hopper_parent_custody import write_once


def compare(root):
    start = time.process_time()
    spec = read(root/'EXECUTION_SPEC.json')
    if (spec['source_sha'] != BASE or mujoco.__version__ != '3.2.7'
            or mujoco.get_mjcb_control() is not None):
        raise ValueError('Expected the original producer and isolated geometry runtime')
    population = read_pin(spec['compiled_population'])
    conditions = {row['id']:row for row in population['conditions']['TRAIN']}
    clusters = {row['id']:row for row in population['clusters']}
    entries = {row['condition']:row for row in read(root/'RECORDS.private.json') if row['split']=='TRAIN'}
    measurements = {}
    with forbid_integration(mujoco) as calls:
        for direction in ('left','right'):
            cell = 'CHV1-TRAIN-F2-PLACE-red-' + direction
            candidate = conditions[cell+'-02']
            scene = clusters[candidate['scene_family']]['scene']
            if candidate['id'] in entries or (root/'cases'/candidate['id']).exists():
                raise ValueError('Expected an unopened destination condition')
            if digest(Path(scene['path'])) != scene['sha256']:
                raise ValueError('Unopened training layout changed')
            model = mujoco.MjModel.from_binary_path(scene['path'])
            rows = []
            for n in ('00','01'):
                case = analyze_case(root, entries[cell+'-'+n], (0,PITCH_DEGREES), world_pitch, model)
                floor = {}
                for angle, stages in case['by_angle_deg'].items():
                    before_close = [values[part]['gap_mm'] for values in stages.values()
                                    for part in ('other_cube','table','selected_nonpad')]
                    closed = case['closing_projection'][angle]
                    floor[angle] = dict(minimum_forbidden_gap_mm=min(*before_close,*closed['forbidden_min_gap_mm'].values()),
                                        bilateral_pad_intersection=closed['both_pads_intersect_red'])
                rows.append(dict(case,summary=floor))
            measurements[direction] = dict(unopened_condition=candidate['id'],scene_sha256=scene['sha256'],rows=rows)
        if calls:
            raise RuntimeError('Unexpected integration during geometric readback')
    return dict(status='SAVED_PLACE_APPROACH_GEOMETRY',source_producer=BASE,
                candidate_world_pitch_deg=PITCH_DEGREES,measurements=measurements,
                admitted_new_episodes=0,model_generations=0,optimizer_updates=0,native_requests=0,
                native_integrations=0,VAL_inputs=0,cpu_seconds=time.process_time()-start)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('original_producer',type=Path)
    cli.add_argument('output',type=Path)
    args = cli.parse_args()
    write_once(args.output,compare(args.original_producer))
