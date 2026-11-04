"""Current FIT action-response diagnosis, no physics or optimizer updates."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-offline',a.output)as spec:
        if spec.get('domain')!='S_CURRENT_FIT_RESPONSE_V1'or spec.get('physics')is not False:raise PermissionError('Offline current-FIT identity required')
        from go2arx5_vla.world_model.response_diagnostics import run
        print(json.dumps(run(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
