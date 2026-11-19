"""Derive the next J screen card from a closed, fully reconciled matched prefix."""
import argparse
import json
from pathlib import Path

from go2arx5_vla.world_model import sla_followup_native_v1 as plan
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import write
from .d416_core_native_launch import cold_owner
from .multitask_panel_suffix import budgets, verified_prefix
from .multitask_stage_cards_v6 import pin, require_unstarted_cases


def prepare(original_config, predecessor, root, authority):
    previous = cold_owner(predecessor)
    original = read_pin(original_config)
    if (previous['scope'] != plan.SCOPE or previous['plan_commit'] != plan.PLAN
            or previous['evaluation_stage'] != plan.JOINT or original['stage'] != plan.JOINT
            or previous['source_sha'] != original['source_sha']
            or previous['source_root'] != original['source_root']
            or previous['compiled_population'] != original['population']
            or previous['interface_predecessor'] != original['E0_interface']
            or set(previous['packages']) != set(plan.JOINT_ARMS)):
        raise ValueError('J_SCREEN_SUFFIX_MUST_KEEP_THE_ORIGINAL_ARMS_POPULATION_AND_SOURCE')
    parent = Path(previous['registered_root'])
    if root.exists() or root.parent.resolve(strict=True) != parent.parent.resolve(strict=True):
        raise ValueError('J_SCREEN_SUFFIX_REQUIRES_A_NEW_SITE_CAPSULE')
    if authority == previous['sim_only_project_authorization']:
        raise PermissionError('J_SCREEN_SUFFIX_REQUIRES_A_NEW_AUTHORIZED_FINITE_WINDOW')
    # The existing launch path checks the fresh authorization, resource headroom
    # and locks. This step leaves the scene registry and runner untouched.
    segments = [*previous.get('previous_panel_segments', []),
                dict(spec=predecessor,result=pin(parent/'RESULT.private.json'))]
    chosen = plan.conditions(read_pin(original['population']),plan.JOINT,plan.JOINT_ARMS)
    carried = verified_prefix(dict(previous,previous_panel_segments=segments),chosen)
    pending = chosen[len(carried):]
    ids = [f"{row['id']}-{arm}" for row,arm in pending]
    require_unstarted_cases(previous['registry'],set(ids))
    requests,ticks,generations = budgets(pending)
    card = dict(original,root=str(root),authority=authority,previous_panel_segments=segments)
    return dict(config=card,carried_science_requests=len(carried),pending_science_requests=requests,
                pending_max_native_ticks=ticks,pending_max_generations=generations,
                first=ids[0],last=ids[-1],new_native_requests=0,new_model_generations=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_config',type=Path)
    parser.add_argument('predecessor_spec',type=Path)
    parser.add_argument('next_root',type=Path)
    parser.add_argument('authority',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = prepare(pin(args.original_config),pin(args.predecessor_spec),
                       args.next_root,pin(args.authority))
    write(args.output,evidence['config'])
    print(json.dumps({key:value for key,value in evidence.items() if key!='config'},allow_nan=False))


if __name__ == '__main__':
    main()
