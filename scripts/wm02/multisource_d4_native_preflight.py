"""Read-only preflight of the sealed D4 DEV panel and transferred model bytes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate_frozen_assets
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.storage import digest
from .multitask_stage_cards_v6 import panel_cards, require_unstarted_cases
from .multitask_teacher_campaign_v6 import interface_records


def inspect(endpoints, population, historical_teacher, gpu_template):
    bundle = d4.portable_endpoints(endpoints)
    bank_pin = dict(path=str(population), sha256=d4.POPULATION_SHA)
    bank = read_pin(bank_pin)
    planned = d4.conditions(bank)
    teacher = json.loads(historical_teacher.read_bytes())
    interface = read_pin(teacher['interface_predecessor'])
    interface_root = Path(interface['registered_root'])
    unit = unit_state('wm02-'+interface_root.name)
    lease = json.loads((interface_root/'control/LEASE.json').read_bytes())
    if (unit['pid'] or unit['cgroup_pids'] or lease['status'] != 'CLOSED'
            or lease['spec_sha256'] != teacher['interface_predecessor']['sha256']):
        raise PermissionError('D4_REQUIRES_COLD_ORIGINAL_MULTITASK_INTERFACE')
    interface_records(interface, teacher['mobile_world_lock'])
    templates = {}
    assets = {}
    for file in interface['cases']:
        case = json.loads(Path(file).read_bytes())
        validate_frozen_assets(case)
        family = case['mobile_condition']['family']
        templates[family] = case
        for path, sha in case['frozen_files'].items():
            if path in assets and assets[path] != sha:
                raise ValueError('CONFLICTING_FROZEN_INTERFACE_ASSET_PINS')
            assets[path] = sha
    for cluster in bank['clusters']:
        scene = cluster['scene']
        assets[scene['path']] = scene['sha256']
    if any(digest(path) != sha for path, sha in assets.items()):
        raise ValueError('FROZEN_D4_NATIVE_ASSET_CHANGED')
    panel = panel_cards(bank, planned, templates, role='STUDENT')
    require_unstarted_cases(teacher['registry'], {r['id'] for r in panel})
    candidate = json.loads(gpu_template.read_bytes())
    historical_site = read_pin(candidate['site_binding'])
    parent = read_pin(candidate['parent_manifest'])
    if (candidate['parent_manifest']['sha256'] != d4.PARENT_SHA or parent['all_files_verified'] is not True
            or Path(candidate['parent_params']).resolve(strict=True)
            != (Path(candidate['parent_manifest']['path']).parent/'model/params').resolve(strict=True)):
        raise ValueError('D4_CORE_FOUNDATION_BYTES_MUST_MATCH_TRAINING')
    return dict(scope=d4.SCOPE, plan_commit=d4.PLAN, compiled_population=bank_pin,
                interface_predecessor=teacher['interface_predecessor'], interface_requests_reused=2,
                native_asset_files_verified=len(assets), core_parent=candidate['parent_manifest'],
                historical_allocation_deadline_epoch=historical_site['allocation_deadline_epoch'],
                total_proposed_worlds=1, matched_pairs=len(panel)//2, effect_caps=dict(native_requests=d4.REQUEST_CAP,
                    native_ticks=d4.TICK_CAP, model_generations=d4.GENERATION_CAP, training_updates=0),
                planned_by_arm=dict(Counter(r['arm'] for r in panel)),
                actual=dict(new_native_requests=0, new_native_ticks=0, new_generations=0, new_updates=0),
                panel_preview=panel, **bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('endpoints', 'population', 'historical_teacher', 'gpu_template'):
        parser.add_argument(key, type=Path)
    print(json.dumps(inspect(**vars(parser.parse_args())), allow_nan=False))


if __name__ == '__main__':
    main()
