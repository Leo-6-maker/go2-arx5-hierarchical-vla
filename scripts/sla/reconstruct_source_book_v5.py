"""Independent ordinary-RGB reconstruction of the actual source BOOK500."""
import json
from pathlib import Path
import argparse

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.runtime_book_v5 import RECIPE,SourceBookEngine,book_probe_indices
from scripts.sla.reconstruct_dense_pilot_v4 import run


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_bytes());package=read_pin(dict(path=str(Path(spec['package']['path'])/'MANIFEST.private.json'),sha256=spec['package']['sha256']))
    if (package['step']!=500 or package['cursor']!=16000 or package['bindings']['recipe']!=RECIPE
            or package['bindings']['source_only']is not True or package['bindings']['target_WBC_qualified']is not False):
        raise ValueError('COMPLETE_BOOK500_SOURCE_ONLY_PACKAGE_REQUIRED')
    result=run(spec,'replay',engine_class=SourceBookEngine,probe_selector=book_probe_indices)
    result.update(source_only=True,target_WBC_qualified=False,target_visits=0,source_visits=0)
    with(Path(spec['registered_root'])/'REPLAY_RESULT.private.json').open('x')as f:json.dump(result,f,indent=2)
