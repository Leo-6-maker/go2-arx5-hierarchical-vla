"""Check one full-posture/query pair; execute only through the original worker."""
import argparse
import json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.fusion_source_posture_paused_query_v3 import episode_limit,PHASE
    spec = json.loads(Path(path).read_text())
    end = episode_limit(spec,json.loads(Path(spec['previous_stage']).read_text()))
    return dict(status='SOURCE_POSTURE_QUERY_V3_CHECKED_ORIGINAL_ADMISSION_STILL_REQUIRED',
        phase=PHASE,max_ticks=end,new_integrations=0,new_queries=0,
        requested_queries=1,returned_VLA_action_installations=0,
        native_execution_authorized=False,native_qualification_claim=False,
        camera_equivalence=False,task_success_claim=False,
        existing_worker='scripts.wm02.pilot1_native_worker')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--check-only',action='store_true')
    args = parser.parse_args(argv)
    value = check(args.spec)
    if not args.execute:
        print(json.dumps(value,sort_keys=True))
        return 0
    from scripts.wm02.pilot1_native_worker import main as original_worker
    return original_worker(['--spec',str(args.spec)])


if __name__ == '__main__':
    raise SystemExit(main())
