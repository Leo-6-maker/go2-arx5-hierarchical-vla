"""Reconstruct the fresh N20 package using ordinary RGB in a fresh process."""
import argparse
import json
from pathlib import Path

from go2arx5_vla.sla_adapter.provider_named_v5 import export
from go2arx5_vla.sla_adapter.runtime_named_v5 import NamedTargetEngine
from scripts.sla.reconstruct_dense_pilot_v4 import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path)
    parser.add_argument('--mode',choices=['export','replay'],required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_bytes())
    result=run(spec,args.mode,engine_class=NamedTargetEngine,export_function=export)
    with (Path(spec['registered_root'])/(args.mode.upper()+'_RESULT.private.json')).open('x') as stream:
        json.dump(result,stream,indent=2)
