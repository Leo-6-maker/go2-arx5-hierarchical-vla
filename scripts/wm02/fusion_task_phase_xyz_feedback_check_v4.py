"""Read-only point1 feedback check; execution stays with the original worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_xyz_feedback_contract_v4 import episode_limit
    spec = json.loads(Path(path).read_bytes())
    end = episode_limit(spec, json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='FINITE_POINT1_XYZ_FEEDBACK_CHECKED_ORIGINAL_ADMISSION_STILL_REQUIRED',
        max_ticks=end, prior_position_and_lateral_failure_retained=True,
        prior_other_seven_non_full_grade_predicates_passed=True,
        runtime_bias_goal_and_feedback_decisions='NOT_CREATED_BY_READ_ONLY_CHECK',
        new_integrations=0, VLA_queries=0, WM_updates=0, Actor_updates=0,
        native_execution_authorized=False, native_qualification_claim=False,
        ordinary_feedback_not_model_benefit=True,
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    result = check(args.spec)
    if not args.execute:
        print(json.dumps(result, sort_keys=True))
        return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec', str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
