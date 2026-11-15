"""Fresh ordinary RGB reconstruction for the separately bound reference labels."""
import argparse,json
from pathlib import Path
from go2arx5_vla.sla_adapter.runtime_reference_v6 import ReferenceTargetEngine
from go2arx5_vla.sla_adapter.provider_reference_v6 import export
from scripts.sla.reconstruct_dense_pilot_v4 import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('--mode',choices=['export','replay'],required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_bytes());result=run(spec,a.mode,engine_class=ReferenceTargetEngine,export_function=export)
    with(Path(spec['registered_root'])/(a.mode.upper()+'_RESULT.private.json')).open('x')as f:json.dump(result,f,indent=2)
