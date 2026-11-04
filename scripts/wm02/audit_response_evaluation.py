"""Audit frozen200 candidate batch/order and original float64 identity."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'core-wm02-response-eval-audit',a.output)as spec:
        from go2arx5_vla.world_model.response_eval_audit import run
        v=run(spec,a.output);print(json.dumps({k:x for k,x in v.items()if k!='models'}))
    return 0
if __name__=='__main__':raise SystemExit(main())
