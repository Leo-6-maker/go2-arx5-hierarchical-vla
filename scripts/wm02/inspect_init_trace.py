"""Read an existing candidate timing trace without native construction or stepping."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.init_trace_reader import summarize
    print(json.dumps(summarize(a.trace),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
