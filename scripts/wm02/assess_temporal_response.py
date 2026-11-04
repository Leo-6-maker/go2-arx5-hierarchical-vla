"""Read saved time diagnostics and apply frozen development gates, no new inference."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.temporal_assessment import assess
    from go2arx5_vla.world_model.storage import write_json
    v=assess(json.loads(a.result.read_text()),json.loads(a.baseline.read_text()));write_json(a.output,v);print(json.dumps(v,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
