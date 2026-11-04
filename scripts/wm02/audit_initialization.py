"""Read SHA-bound saved target initialization logs, without starting target components."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.initialization_audit import audit
    result=audit(a.manifest,a.output)
    print(json.dumps({k:v for k,v in result.items()if k!='episodes'}));return 0
if __name__=='__main__':raise SystemExit(main())
