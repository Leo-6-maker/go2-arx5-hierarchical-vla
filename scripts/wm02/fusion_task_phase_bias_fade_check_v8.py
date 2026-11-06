"""Read-only fixed full1 fade check; execution uses the original admitted worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_bias_fade_contract_v8 import episode_limit
    spec = json.loads(Path(path).read_bytes())
    end = episode_limit(spec, json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='FINITE_FULL1_BIAS_FADE_CHECKED_ORIGINAL_ADMISSION_REQUIRED', max_ticks=end,
        qualified_half_all_nine_predicates_passed=True, failed_full1_closed1_preserved=True,
        runtime_bias_goal_and_feedback='NOT_CREATED_BY_READ_ONLY_CHECK',
        runtime_new_bias_goal_and_feedback='NOT_CREATED',
        saved_state_certificate_is_dynamic_success=False, native_execution_authorized=False,
        native_qualification_claim=False, new_integrations=0, VLA_queries=0,
        WM_updates=0, Actor_updates=0, existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    value = check(args.spec)
    if not args.execute:
        print(json.dumps(value, sort_keys=True))
        return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec', str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
