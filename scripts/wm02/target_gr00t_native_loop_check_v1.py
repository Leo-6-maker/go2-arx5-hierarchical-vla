"""Read-only paired GR00T loop admission; execution uses the original worker."""
import argparse,json
from pathlib import Path


def check(path):
    from go2arx5_vla.world_model.target_gr00t_native_loop_v1 import episode_limit
    s=json.loads(Path(path).read_bytes())
    end=episode_limit(s,json.loads(Path(s['previous_stage']).read_bytes()))
    return dict(status='TARGET_GR00T_PREFIX_PAIR_CHECKED_NOT_EXECUTED',max_ticks=end,
        registered_final_step=end,new_integrations=0,new_queries=0,new_updates=0,
        native_execution_authorized=False,data_role=s['data_role'],data_parent=s['data_parent'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    result=check(a.spec)
    if not a.execute:print(json.dumps(result));return 0
    from scripts.wm02.pilot1_native_worker import main as worker
    return worker(['--spec',str(a.spec)])


if __name__=='__main__':raise SystemExit(main())
