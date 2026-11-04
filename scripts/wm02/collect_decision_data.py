"""Collect a predeclared40-parent source-only physical decision dataset, no model training."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-native',a.output,physics=True)as spec:
        if spec.get('domain')!='S_NEW_DECISION_DATA_V5'or spec.get('target_physics')is not False:
            raise PermissionError('New bounded source-only identity required')
        from go2arx5_vla.world_model.decision_collection import collect
        print(json.dumps(collect(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
