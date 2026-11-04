"""Finite M4-B fresh matched200 conditional interaction experiment."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-gpu0-real-s',a.output,gpu=True)as spec:
        if spec.get('authorization_scope')!='USER_WM02_AUTONOMOUS_MECHANISM_MODEL_ONLY_20261104'or spec.get('mechanism_candidate')!='M4_B':raise PermissionError('Exact delegated M4-B scope required')
        from go2arx5_vla.world_model.temporal_training import run
        from go2arx5_vla.world_model.absolute_joint_training import update
        print(json.dumps(run(spec,a.output,update_fn=update)))
    return 0
if __name__=='__main__':raise SystemExit(main())
