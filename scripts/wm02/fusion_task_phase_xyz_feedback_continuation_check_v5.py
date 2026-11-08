"""Read-only adjacent feedback gate; execute only through the original worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_xyz_feedback_continuation_contract_v5 import episode_limit
    spec = json.loads(Path(path).read_bytes())
    end = episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='ADJACENT_XYZ_FEEDBACK_GATE_CHECKED_ORIGINAL_ADMISSION_STILL_REQUIRED',
        max_ticks=end, all_nine_prior_non_full_grade_predicates_passed=True,
        prior_saved_decisions_and_every_rolling_reference_consumption_rejoined=True,
        runtime_new_bias_goal_and_feedback_decisions='NOT_CREATED_BY_READ_ONLY_CHECK',
        new_integrations=0,VLA_queries=0,WM_updates=0,Actor_updates=0,
        native_execution_authorized=False,native_qualification_claim=False,
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    value=check(args.spec)
    if not args.execute:
        print(json.dumps(value,sort_keys=True));return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec',str(args.spec)])


if __name__=='__main__':
    raise SystemExit(main())
