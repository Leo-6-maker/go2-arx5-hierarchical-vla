"""Read-only C02 finite feedback initializer contract check; execute only via original worker."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);g=p.add_mutually_exclusive_group();g.add_argument('--check-only',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    from go2arx5_vla.world_model.fusion_c02_source_state_feedback_v2 import episode_limit,PHASE
    s=json.loads(a.spec.read_bytes());end=episode_limit(s,json.loads(Path(s['previous_stage']).read_bytes()))
    if not a.execute:
        print(json.dumps(dict(status='C02_FEEDBACK_CONTRACT_CHECKED_NOT_EXECUTED',phase=PHASE,max_ticks=end,new_integrations=0,new_queries=0,new_updates=0,native_execution_authorized=False)));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(a.spec)])
if __name__=='__main__':raise SystemExit(main())
