"""Export sealed actual target demonstrations; no physics or model training."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle-kind',choices=('FORMAL','FIT4_DIAGNOSTIC'),default='FORMAL')
    p.add_argument('--fit4-selection',type=Path);a=p.parse_args(argv)
    from go2arx5_vla.world_model.target_gr00t_dataset_v1 import build
    print(json.dumps(build(a.panel,a.runs_root,a.output,bundle_kind=a.bundle_kind,fit4_selection_path=a.fit4_selection)));return 0

if __name__=='__main__':raise SystemExit(main())
