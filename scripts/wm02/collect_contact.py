"""Finite registered four-process S contact identification; never target physics."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-native',a.output,physics=True)as spec:
        if spec.get('domain')!='S_CONTACT_IDENTIFICATION_V1'or spec.get('target_physics')is not False:raise PermissionError('Explicit source-only identity required')
        from go2arx5_vla.world_model.contact_identification import collect
        print(json.dumps(collect(spec,a.output)))
    return 0

if __name__=='__main__':raise SystemExit(main())
