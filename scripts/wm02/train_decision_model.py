"""Matched new V5 source dynamics, fixed200 updates per arm, no Actor optimizer."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-gpu1-real-s',a.output,gpu=True)as spec:
        if spec['data_domain']!='WM02_DECISION_PAIRED_REPLAY_V5':raise PermissionError('Separate V5 data identity required')
        from go2arx5_vla.world_model.contact_training import run
        from go2arx5_vla.world_model.decision_training import read,evaluate
        print(json.dumps(run(a.output,spec,reader=read,evaluator=evaluate,protocol='WM02_NEW_DECISION_DATA_V5',evaluate_dev=True)))
    return 0
if __name__=='__main__':raise SystemExit(main())
