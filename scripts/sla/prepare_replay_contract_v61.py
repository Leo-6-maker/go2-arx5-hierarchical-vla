"""Pin S16/RP8 inputs from the already verified V6 manifest;no model/physics."""
import argparse,hashlib,json
from pathlib import Path
from go2arx5_vla.sla_adapter.action_contract_v61 import DATA_SHA,STATS_SHA,selection


def run(dataset,output):
    dataset,output=Path(dataset),Path(output);p=dataset/'MANIFEST.private.json'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=DATA_SHA:
        raise ValueError('EXACT_V6_REFERENCE_DATASET_BYTES_REQUIRED')
    m=json.loads(p.read_bytes());stats=dataset/m['statistics']['file']
    if hashlib.sha256(stats.read_bytes()).hexdigest()!=STATS_SHA:
        raise ValueError('EXISTING_FULL_N20_TRAIN_STATISTICS_REQUIRED')
    value=selection(m)
    with output.open('x')as f:json.dump(value,f,indent=2,allow_nan=False)
    return dict(selection_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        windows=16,RP8_conditions=8,layouts=value['selected_layouts'],model_loads=0,optimizer_updates=0,physical_requests=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.dataset,a.output)))
