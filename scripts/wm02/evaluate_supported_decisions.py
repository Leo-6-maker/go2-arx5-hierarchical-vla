"""Offline correction using real native support and fixed200 models; no optimizer."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-offline',a.output)as spec:
        if spec.get('domain')!='S_FROZEN_SUPPORT_RESCORE_V2'or spec.get('physics')is not False:raise PermissionError('Explicit offline correction required')
        from go2arx5_vla.world_model.support_evaluation import evaluate
        print(json.dumps(evaluate(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
