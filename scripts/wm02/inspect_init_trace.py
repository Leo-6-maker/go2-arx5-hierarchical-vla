"""Read an existing candidate timing trace without native construction or stepping."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--audit',action='store_true');p.add_argument('--max-native-ticks',type=int);a=p.parse_args(argv)
    from go2arx5_vla.world_model.init_trace_reader import summarize,audit_trace
    print(json.dumps(audit_trace(a.trace,max_native_ticks=a.max_native_ticks)if a.audit else summarize(a.trace),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
