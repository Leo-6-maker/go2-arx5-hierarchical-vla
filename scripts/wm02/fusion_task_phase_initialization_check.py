"""Read-only task-phase recipe check; execution delegates to the original worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_initialization_contract import episode_limit
    spec = json.loads(Path(path).read_text())
    end = episode_limit(spec, json.loads(Path(spec['previous_stage']).read_text()))
    return dict(status='TASK_PHASE_RECIPE_CHECKED_ORIGINAL_ADMISSION_STILL_REQUIRED',
        max_ticks=end, new_integrations=0, VLA_queries=0, VLA_action_installations=0,
        native_execution_authorized=False, native_qualification_claim=False,
        source_pose_is_expert_action_label=False, object_and_goal_are_policy_inputs=False,
        task_success_claim=False, runtime_goal_binding='NOT_CREATED_BY_READ_ONLY_CHECK',
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    value = check(args.spec)
    if not args.execute:
        print(json.dumps(value, sort_keys=True))
        return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec', str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
