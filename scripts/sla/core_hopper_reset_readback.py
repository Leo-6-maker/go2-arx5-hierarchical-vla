"""Compare simulator-time-zero states across the actual expert instructions in each layout."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import time

from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned


def double_hash(values):
    if not all(map(math.isfinite, values)):
        raise ValueError('Nonfinite native state')
    return sha256(struct.pack(f'<{len(values)}d', *values)).hexdigest()


def native_reset(case, condition, cluster, source, geometry_cache):
    spec = json.loads((case/'EXECUTION_SPEC.json').read_bytes())
    scene = cluster['scene']
    mobile = spec['mobile_condition']
    if (spec['panel_entry_id'] != case.name or spec['model_path'] != scene['path']
            or spec['task_instruction_id'] != condition['instruction'] or spec['source_sha'] != source
            or spec['frozen_files'][scene['path']] != scene['sha256']
            or spec['multitask_geometry'] != cluster['scoring_geometry']
            or (mobile['layout'],mobile['reset'],mobile['seed']) !=
               (cluster['id'],cluster['reset'],cluster['scene_seed'])):
        raise ValueError('Case geometry or registered reset differs: ' + case.name)
    if scene['path'] not in geometry_cache:
        geometry_cache[scene['path']] = digest(Path(scene['path']))
    if geometry_cache[scene['path']] != scene['sha256']:
        raise ValueError('Frozen numerical scene changed')
    native = case/'native-01'
    startup = native/'initialization.jsonl'
    raw = startup.read_bytes()
    state = json.loads(raw)
    with (native/'physics.jsonl').open('rb') as trace:
        header = json.loads(trace.readline())
        first = json.loads(trace.readline())
    marker = json.loads((native/'BEFORE_FIRST_INTEGRATION.json').read_bytes())
    abi = cluster['scoring_geometry']['expected_ABI']
    if (state['kind'] != 'actual_startup_V1' or state['time'] != 0 or marker['sim_time_s'] != 0
            or header['kind'] != 'model' or first['kind'] != 'sample' or first['step'] != 1
            or first['sim_time_s'] != header['timestep'] or header['timestep'] != .002):
        raise ValueError('Missing time-zero or first integration evidence: ' + case.name)
    for name, field in [('nq','qpos'),('nv','qvel'),('nu','ctrl')]:
        if header[name] != abi[name] or len(state[field]) != abi[name]:
            raise ValueError('Observed state dimensions differ: ' + case.name)
    joints = {joint['name']:joint for joint in header['joints']}
    xyz = {}
    for color, obj in cluster['scoring_geometry']['objects'].items():
        joint = joints[obj['joint']]
        idx = joint['qposadr']
        xyz[color] = state['qpos'][idx:idx+3]
        if joint['type'] != 0 or xyz[color] != cluster['card']['object_positions_world'][color]:
            raise ValueError('Startup object position differs from registered scene: ' + case.name)
    full_state = double_hash(state['qpos']+state['qvel']+state['ctrl'])
    return dict(startup_file_sha256=sha256(raw).hexdigest(),initial_state_sha256=full_state,object_xyz=xyz)


def audit(catalog, config_sha, count):
    cfg = pinned(dict(path=str(catalog/'CONFIG.json'),sha256=config_sha))
    if cfg['plan'] != PLAN or cfg['population']['sha256'] != POP:
        raise ValueError('Unregistered population')
    registered = pinned(cfg['population'])
    population = {c['id']:c for split in ('TRAIN','VAL') for c in registered['conditions'][split]}
    clusters = {c['id']:c for c in registered['clusters']}
    receipt = json.loads((catalog/'catalog'/f'TRAIN-{count:03d}.json').read_bytes())
    seal = pinned(receipt)
    if (seal['plan'] != PLAN or seal['split'] != 'TRAIN' or seal['count'] != count
            or seal['episode_sha256'] != receipt['membership'] or len(set(receipt['membership'])) != count):
        raise ValueError('Unexpected snapshot identity')
    folder = Path(receipt['path']).parent
    manifest = pinned(dict(path=str(folder/'MANIFEST.private.json'),sha256=seal['manifest_sha256']))
    membership = {r['id']:(s,r) for s,r in zip(seal['episode_sha256'],manifest['episodes'],strict=True)}
    if len(membership) != count or sum(r['windows'] for _,r in membership.values()) != seal['windows']:
        raise ValueError('Snapshot window or episode count differs')
    geometry_cache, cases, unstarted, logs = {}, {}, [], {}
    grouped = defaultdict(list)
    for spec_sha, dirname in cfg['producers'].items():
        owner = Path(dirname)
        spec = pinned(dict(path=str(owner/'EXECUTION_SPEC.json'),sha256=spec_sha))
        ledger = owner/'RECORDS.private.json'
        if not ledger.is_file():
            continue
        raw = ledger.read_bytes()
        logs[spec_sha] = sha256(raw).hexdigest()
        for row in json.loads(raw):
            ident = row['condition']
            cond = population[ident]
            case = owner/'cases'/ident
            terminal = pinned(row['result'])
            if (ident in cases or ident in unstarted or row['split'] != cond['split']
                    or spec['compiled_population'] != cfg['population']
                    or row['result']['path'] != str(case/'RESULT.private.json')
                    or terminal['source_sha'] != spec['source_sha']
                    or terminal['condition']['collection_episode'] != ident
                    or terminal['physical_requests'] != row['requests']
                    or terminal['actual_native_integrations'] != row['ticks']):
                raise ValueError('Owner, population or terminal differs: ' + ident)
            if not row['requests']:
                if row['ticks'] or row['admitted']:
                    raise ValueError('Unrequested condition has native results')
                unstarted.append(ident)
                continue
            if row['requests'] != 1:
                raise ValueError('Invalid native request count')
            cluster = clusters[cond['cluster']]
            if (cond['reset'],cond['scene_seed'],cond['family'],cond['split']) != (
                    cluster['reset'],cluster['scene_seed'],cluster['family'],cluster['split']):
                raise ValueError('Condition and layout differ')
            observation = native_reset(case,cond,cluster,spec['source_sha'],geometry_cache)
            if ident in membership:
                fingerprint, ep = membership[ident]
                if (not row['admitted'] or row['episode']['sha256'] != fingerprint or cond['split'] != 'TRAIN'
                        or ep['scene_sha256'] != cluster['scene']['sha256'] or ep['reset'] != cluster['reset']
                        or ep['task'] != cond['instruction'] or ep['native_result_sha256'] != row['result']['sha256']):
                    raise ValueError('Published episode differs from the measured request')
            cases[ident] = dict(**observation,scene=cluster['id'],instruction=cond['instruction'],
                split=cond['split'],terminal_sha256=row['result']['sha256'],admitted=row['admitted'],
                sealed_windows=membership[ident][1]['windows'] if ident in membership else 0)
            grouped[cluster['id']].append(cases[ident])
    if membership.keys() - cases.keys():
        raise ValueError('Missing native terminal for sealed episode')
    scenes = {}
    for key, rows in sorted(grouped.items()):
        instructions = sorted(r['instruction'] for r in rows)
        if len(instructions) != len(set(instructions)):
            raise ValueError('Repeated instruction in layout ' + key)
        scenes[key] = dict(split=clusters[key]['split'],scene_sha256=clusters[key]['scene']['sha256'],
            reset=clusters[key]['reset'],scene_seed=clusters[key]['scene_seed'],instructions=instructions,
            state_sha256=sorted({r['initial_state_sha256'] for r in rows}),
            admitted=sum(r['admitted'] for r in rows),sealed=sum(bool(r['sealed_windows']) for r in rows))
    tasks = Counter(ep['task'] for _,ep in membership.values())
    return dict(plan=PLAN,observed_epoch=time.time(),population=POP,catalog_config=config_sha,
        producer_ledger_hashes=logs,seal=receipt['sha256'],sealed_episodes=count,sealed_windows=seal['windows'],
        sealed_scenes=len({cases[k]['scene'] for k in membership}),sealed_instruction_counts=dict(tasks),
        zero_request_terminals=unstarted,scenes=scenes,cases=cases,
        optimizer_updates=0,model_generations=0,new_native_requests=0,new_integrations=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('catalog',type=Path)
    parser.add_argument('catalog_sha')
    parser.add_argument('count',type=int)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    write_once(args.output,audit(args.catalog,args.catalog_sha,args.count))
