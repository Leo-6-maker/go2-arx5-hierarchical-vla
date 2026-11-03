"""Read an explicit P0 owner observation into one immutable WM02 specification."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.resolution import resolve
    from go2arx5_vla.world_model.storage import write_json,digest
    result=resolve(json.loads(a.binding.read_text()));result['binding_sha256']=digest(a.binding)
    write_json(a.output,result)
    print(json.dumps(dict(status=result['target_status'],blockers=result['target_blockers'],output=str(a.output))))
    return 0


if __name__=='__main__':raise SystemExit(main())
