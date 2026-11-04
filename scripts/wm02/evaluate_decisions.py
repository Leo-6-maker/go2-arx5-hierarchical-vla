"""Finite registered CPU evaluation of frozen WM checkpoints; zero updates and physics."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-frozen-model-cpu',a.output)as spec:
        if spec.get('physics')is not False or spec.get('WM_updates_allowed')!=0:raise PermissionError('Frozen zero-effect CPU inference only')
        from go2arx5_vla.world_model.decision_diagnostics import evaluate
        print(json.dumps(evaluate(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
