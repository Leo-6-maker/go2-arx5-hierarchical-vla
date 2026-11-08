"""Read-only finite continuation gate; execution still uses original admission."""
import argparse,json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_task_phase_pose_continuation_contract_v7 import episode_limit
    spec=json.loads(Path(path).read_bytes());end=episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='FINITE_GRADED_POSE_GATE_CHECKED_ORIGINAL_ADMISSION_REQUIRED',max_ticks=end,
        native_execution_authorized=False,native_qualification_claim=False,
        runtime_new_bias_goal_and_feedback='NOT_CREATED',new_integrations=0,VLA_queries=0,WM_updates=0,Actor_updates=0,
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv);value=check(args.spec)
    if not args.execute:print(json.dumps(value,sort_keys=True));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(args.spec)])


if __name__=='__main__':raise SystemExit(main())
