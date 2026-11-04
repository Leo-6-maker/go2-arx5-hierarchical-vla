"""M1/M2 fixed200 comparison on existing real V5 data; zero Actor or physics."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv);binding=json.loads(a.spec.read_text())
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,binding['profile'],a.output,gpu=True)as spec:
        if spec.get('authorization_scope')!='USER_WM02_RESPONSE_M1_M2_MODEL_ONLY_20261104':raise PermissionError('Separate response comparison identity required')
        from go2arx5_vla.world_model.response_training import run
        print(json.dumps(run(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
