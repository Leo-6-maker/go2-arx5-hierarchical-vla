"""Prospective8-parent source experiment only after a SHA-pinned joint DEV pass."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('spec','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-native',a.output,physics=True)as spec:
        if spec.get('domain')not in('S_PROSPECTIVE_B2_NEW8_V1','S_PROSPECTIVE_B2_COAST_CLOSE_V1')or spec.get('target_physics')is not False:raise PermissionError('New S-only prospective validation identity required')
        from go2arx5_vla.world_model.new_s_validation import run
        print(json.dumps(run(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
