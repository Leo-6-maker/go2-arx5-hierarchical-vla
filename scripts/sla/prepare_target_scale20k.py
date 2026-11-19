"""Freeze the SCALE20K holdouts and admit known, complete historical PLACE."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.dense_g7_v6 import VERSION
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_MANIFEST, C20_STATISTICS
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import PLAN
from go2arx5_vla.sla_adapter.scale20k_target import SCHEMA
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS
from scripts.pi0292.durable import read, write
from scripts.sla.export_multitask_g7_v6 import fp32_audit
from scripts.sla.multitask_g7_capture_v6 import copy_shards, read_capture
from scripts.wm02.multitask_d2_population import position_id
from scripts.wm02.multitask_population_v6 import compile_population

OFFSETS_MM = {'SCALE_DEV': ((6, -16), (-14, 12)), 'SCALE_CONFIRM': ((15, -9), (-12, -15))}
PLACE_IDS = ('M-TEACHER-F1-0-PLACE-blue-left', 'M-TEACHER-F1-1-PLACE-red-right',
             'M-TEACHER-F1-1-PLACE-blue-left')


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def geometry(row):
    return row['family'], position_id(row['card'])


def holdout_population(cards, historical):
    occupied = {geometry(row) for bank in historical for row in bank['clusters']}
    resets = {row['reset'] for bank in historical for row in bank['clusters']}
    clusters, conditions = [], {}
    for stage_index, (stage, offsets) in enumerate(OFFSETS_MM.items()):
        conditions[stage] = []
        for index, (dx, dy) in enumerate(offsets):
            for family_index, family in enumerate(('F1', 'F2')):
                card = deepcopy(cards[family])
                for position in card['object_positions_world'].values():
                    position[0] += dx/1000
                    position[1] += dy/1000
                layout, reset = f'{stage}-{family}-{index+1}', 720000+1000*stage_index+10*index+family_index
                row = dict(id=layout, stage=stage, family=family, card=card, reset=reset,
                           collection_lineage=f'SCALE20K-20261119:{stage}')
                if geometry(row) in occupied or reset in resets:
                    raise ValueError('SCALE20K_HOLDOUT_REUSES_FROZEN_LAYOUT_OR_RESET')
                occupied.add(geometry(row))
                resets.add(reset)
                clusters.append(row)
                for index_, instruction in enumerate(INSTRUCTIONS):
                    conditions[stage].append(dict(id=f'{layout}-{instruction}', cluster=layout,
                        instruction=instruction, family=family, reset=reset,
                        noise_seed=2026111900+16*reset+index_))
    return dict(schema='SLA_SCALE20K_FROZEN_TARGET_EVALUATION_V1', plan=PLAN,
                conditions=conditions, clusters=clusters, scene_instruction_independent=True,
                old_TEST_reused=False, native_dispatch_qualified=False)


def heldout_fingerprints(banks, legacy_dev, extra_scene_sets):
    result = dict(scenes=set(), geometry=set(), lineages=set())
    for bank in banks:
        for row in bank['clusters']:
            if row['stage'] == 'TEACHER':
                continue
            result['geometry'].add(geometry(row))
            result['scenes'].add(row['scene']['sha256'])
            if row.get('collection_lineage'):
                result['lineages'].add(row['collection_lineage'])
    for row in legacy_dev:
        result['lineages'].update((row['scene_lineage'], row['session']))
    for rows in extra_scene_sets:
        result['scenes'].update(row['scene']['sha256'] for row in rows)
        result['lineages'].update(row['collection_lineage'] for row in rows if row.get('collection_lineage'))
    return result


def prepare(spec):
    start = time.monotonic()
    root = Path(spec['registered_root'])
    if (spec['plan'] != PLAN or not root.is_dir() or (root/'MANIFEST.private.json').exists()
            or spec['deadline_epoch'] <= time.time()+60
            or tuple(candidate['id'] for candidate in spec['place_candidates']) != PLACE_IDS):
        raise ValueError('SCALE20K_EXACT_HISTORICAL_CANDIDATE_AND_COLD_SPEC_REQUIRED')
    bank_pins = spec['historical_populations']
    banks = [read_pin(member) for member in bank_pins]
    v6 = banks[0]
    teachers = {row['id']:row for row in v6['conditions']['TEACHER']}
    scenes = {row['id']:row for row in v6['clusters']}
    c20 = read_pin(spec['C20_manifest'])
    if spec['C20_manifest']['sha256'] != C20_MANIFEST or c20['statistics']['sha256'] != C20_STATISTICS:
        raise ValueError('ORIGINAL_C20_TRAIN_STATISTICS_MUST_STAY_FROZEN')
    stat_path = Path(spec['C20_manifest']['path']).parent/c20['statistics']['file']
    stats = read_pin(dict(path=str(stat_path), sha256=C20_STATISTICS))
    statistics = {name:Quantiles(stats[name]['q01'],stats[name]['q99']) for name in ('state','actions')}
    legacy = read_pin(spec['C20_population'])
    c20_scenes = read_pin(spec['C20_scene_catalog'])
    protected_layouts = {(row['family'],row['layout_id']) for row in legacy['dev16']}
    legacy_dev_scenes = [row for row in c20_scenes if (row['family'],row['layout_id']) in protected_layouts]
    if len(legacy_dev_scenes) != len(protected_layouts):
        raise ValueError('ORIGINAL_C20_DEV_SCENE_METADATA_REQUIRED')
    legacy_dev_scenes = [dict(scene=dict(sha256=row['sha256'])) for row in legacy_dev_scenes]
    v52 = read_pin(spec['V52_split'])
    extra_scene_sets = [v52['layouts'], legacy_dev_scenes]
    proposed = holdout_population(spec['prototype_cards'], banks)
    freeze = compile_population(spec, proposed=proposed)
    holdout_pin = pin(root/'COMPILED_POPULATION.private.json')
    heldouts = read_pin(holdout_pin)
    original = heldout_fingerprints(banks, legacy['dev16'], extra_scene_sets)
    if any(row['scene']['sha256'] in original['scenes'] for row in heldouts['clusters']):
        raise ValueError('SCALE20K_HOLDOUT_SCENE_BYTES_DUPLICATE_AN_ARCHIVE')
    protected = heldout_fingerprints([*banks,heldouts], legacy['dev16'], extra_scene_sets)
    split_path = root/'SPLIT_INVENTORY.private.json'
    write(split_path, dict(schema='SLA_SCALE20K_TARGET_SPLIT_INVENTORY_V1',
        original_populations=bank_pins, C20_population=spec['C20_population'], C20_scene_catalog=spec['C20_scene_catalog'],
        V52_split=spec['V52_split'], new_evaluation_population=holdout_pin,
        protected_scene_hashes=sorted(protected['scenes']),
        protected_collection_lineages=sorted(protected['lineages']), old_TEST_rollouts_read=False,
        C20_TRAIN_layouts=sorted({(row['family'],row['layout']) for row in c20['episodes']}),
        new_layout_count=freeze['compiled_clusters']))
    rows, episodes = [], []
    for member in spec['place_candidates']:
        cid = member['id']
        condition = teachers[cid]
        scene = scenes[condition['cluster']]
        case = read_pin(member['case'])
        result = read_pin(member['native'])
        native_root = Path(case['registered_root'])
        lease = read(native_root/'control/LEASE.json')
        source_lineage = result['condition']['scene_lineage']
        collection_lineage = f"{bank_pins[0]['sha256']}:TEACHER"
        row = dict(id=cid, task=condition['instruction'], family=condition['family'],
            scene_sha256=scene['scene']['sha256'], scene_template_lineage=source_lineage,
            collection_lineage=collection_lineage, result_sha256=member['native']['sha256'],
            admitted=False)
        if (case['sim_only_role'] != 'EXPERT' or case['panel_entry_id'] != cid
                or case['collection_contract'] != 'SLA_V6_MULTITASK_TEACHER_CURRENT_FULL_H40_AND_CONSUMED_MASK'
                or case['mobile_condition']['layout'] != condition['cluster']
                or case['mobile_condition']['reset'] != condition['reset']
                or case['mobile_condition']['seed'] != condition['noise_seed']
                or case['frozen_files'][case['model_path']] != row['scene_sha256']
                or scene['stage'] != 'TEACHER' or lease['status'] != 'CLOSED'
                or lease['spec_sha256'] != member['case']['sha256'] or result['source_sha'] != case['source_sha']
                or result['physical_requests'] != 1 or result['optimizer_updates'] or result['model_generations']
                or Path(member['native']['path']) != native_root/'RESULT.private.json'):
            row['first_failure'] = 'ORIGINAL_CLOSED_EXPERT_IDENTITY_NOT_PROVEN'
        elif (row['scene_sha256'] in protected['scenes'] or geometry(scene) in protected['geometry']
                or collection_lineage in protected['lineages'] or source_lineage in protected['lineages']):
            row['first_failure'] = 'PROTECTED_SCENE_OR_COLLECTION_LINEAGE_OVERLAP'
        else:
            try:
                capture, ledger, frame, codec, arrays = read_capture(condition,case,result)
                audit = fp32_audit([(cid,codec,arrays,arrays['previous_actual_command'])],statistics)
                groups = audit['label_margin_by_domain_codec_phase']
                if any(g['over_original_gate'] or (g['codec_path']=='fp32_codec' and g['over_margin']) for g in groups):
                    raise ValueError('ORIGINAL_C20_FP32_TRAINING_MARGIN_FAILED')
                folder = root/cid
                folder.mkdir()
                shards = copy_shards(capture,ledger,arrays,folder)
                labels = folder/'LABELS.npz'
                with labels.open('xb') as stream:
                    np.savez(stream,**arrays)
                windows = ledger['queries']
                row.update(admitted=True, windows=windows, confirmed_P10_atoms=10*windows,
                    historical_native_ticks=result['actual_native_integrations'],
                    score_events=result['detail']['score']['events'],
                    FP32_maximum_error=audit['maximum_FP32_roundtrip_error'])
                episodes.append(dict(id=cid, split='TRAIN', task=row['task'], family=row['family'],
                    language=case['language'], windows=windows, mount=dict(T_policy_from_mount=frame.tolist()),
                    expert_stage_audit_only=arrays['phase'].tolist(),
                    labels=dict(file=str(labels.relative_to(root)),sha256=sha(labels)),
                    frame_shards=shards, native_result=member['native'], individual_pass=True))
            except (ValueError, KeyError, OSError) as error:
                row['first_failure'] = str(error)
        rows.append(row)
    windows = sum(row['windows'] for row in episodes)
    report_path = root/'ACTUAL.private.json'
    write(report_path,dict(status='SCALE20K_TARGET_DATA_FROZEN', cases=rows,
        protected_scene_count=len(protected['scenes']), protected_geometry_count=len(protected['geometry']),
        C20_TRAIN_episodes=len(c20['episodes']), C20_TRAIN_windows=c20['windows'],
        PLACE_TRAIN_episodes=len(episodes), PLACE_TRAIN_windows=windows,
        PLACE_distinct_scenes=len({row['scene_sha256'] for row in rows if row['admitted']}),
        PLACE_collection_lineages=len({row['collection_lineage'] for row in rows if row['admitted']}),
        old_TEST_rollouts_read=False, new_optimizer_updates=0, new_model_generations=0,
        new_native_requests=0, new_native_ticks=0, wall_s=time.monotonic()-start))
    path = root/'MANIFEST.private.json'
    write(path,dict(schema=SCHEMA, state='committed', split='TRAIN', representation=VERSION,
        C20_manifest_sha256=C20_MANIFEST, C20_statistics_sha256=C20_STATISTICS,
        split_inventory=pin(split_path), audit=pin(report_path), windows=windows,
        total_windows=c20['windows']+windows, episodes=episodes, phase_input_to_student=False,
        teacher_memory_input=False, unconsumed_suffix_is_executed=False))
    return dict(admission=pin(path), audit=pin(report_path), population=holdout_pin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    print(json.dumps(prepare(read_pin(dict(path=args.spec,sha256=args.sha256)))))
