"""Run frozen H1-H16 response/baseline diagnosis, no optimizer or physics."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'core-wm02-temporal-diagnostic',a.output)as spec:
        from go2arx5_vla.world_model.temporal_diagnostics import run
        v=run(spec,a.output);print(json.dumps({k:v[k]for k in('source_sha','data_sha256','new_physics','WM_updates','Actor_updates','CPU_s','wall_s')}))
    return 0
if __name__=='__main__':raise SystemExit(main())
