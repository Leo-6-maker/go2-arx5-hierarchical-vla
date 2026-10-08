"""Current scientific authority; legacy runtime ABI release remains separately named."""
import json,time,os
from pathlib import Path
from budget import publication

def load(root):
    root=Path(root);publication(root)
    v=json.loads((root/'design/RUNTIME_AUTHORITY.json').read_text())
    assert v['formal_cap']==4096 and v['all_updates_cap']==4208 and v['canary_cap']==32 and v['technical_cap']==64
    assert v['deadline_epoch'] is None
    return v

def allowed(root,kind,reserve_s=0):
    root=Path(root);v=load(root)
    if kind=='update':
        route=json.loads((root/'design/TRAINING_ROUTE.json').read_text())
        assert route['arms']==['CTRL2048','LR5X2048'] and route['K']==4
        allocation=root/'runtime/TRAINING_ALLOCATION.json'
        if allocation.exists():assert time.time()-json.loads(allocation.read_text())['started_epoch']+reserve_s<v['training_GPU_reservation_s']
    elif kind in ('query','episode'):
        from budget import time_allowed
        time_allowed(root,reserve_s)
    return v
