"""Frozen B0 mechanism diagnostics only: zero optimizer, source or target integration."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'core-wm02-joint-mechanism',a.output)as spec:
        from go2arx5_vla.world_model.mechanism_diagnostics import run
        result=run(spec,a.output);print(json.dumps({k:result[k]for k in('status','source_sha','data_sha256','WM_updates','new_physics','CPU_s','wall_s')}))
    return 0
if __name__=='__main__':raise SystemExit(main())
