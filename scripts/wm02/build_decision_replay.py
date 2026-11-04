"""Read-only export of the original four-DEV contact probe."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','manifest','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.decision_diagnostics import export_onset
    print(json.dumps(export_onset(a.root,a.manifest,a.output)));return 0
if __name__=='__main__':raise SystemExit(main())
