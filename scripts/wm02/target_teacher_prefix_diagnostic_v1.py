"""Check or execute one newly registered privileged two-prefix diagnostic."""
import argparse,json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.target_teacher_prefix_diagnostic_v1 import episode_limit
    spec=json.loads(Path(path).read_bytes());end=episode_limit(spec,json.loads(Path(spec['previous_stage']).read_bytes()))
    return dict(status='TEACHER_TWO_PREFIX_CHECKED_NOT_EXECUTED',max_ticks=end,
        new_integrations=0,new_queries=0,new_updates=0,native_execution_authorized=False,
        expert_training_eligible=False,GR00T_native_interface_qualified=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv);result=check(a.spec)
    if not a.execute:print(json.dumps(result));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(a.spec)])


if __name__=='__main__':raise SystemExit(main())
