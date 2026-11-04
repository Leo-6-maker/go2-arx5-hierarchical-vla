"""Matched new V5 source dynamics, fixed200 updates per arm, no Actor optimizer."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    profile=json.loads(a.spec.read_text())['profile']
    if profile not in('wm02-gpu1-real-s','wm02-gpu0-real-s'):raise PermissionError('Explicit authorized model-only GPU identity required')
    binding=json.loads(a.spec.read_text())
    from go2arx5_vla.world_model.storage import digest
    if digest(binding['support_replay'])!=binding['support_replay_sha256']:raise ValueError('Actual support evaluation derivative required before GPU start')
    with admitted_run(a.spec,a.run_id,profile,a.output,gpu=True)as spec:
        if spec['data_domain']!='WM02_DECISION_PAIRED_REPLAY_V5':raise PermissionError('Separate V5 data identity required')
        from go2arx5_vla.world_model.contact_training import run
        from go2arx5_vla.world_model.decision_training import read,evaluate
        def reader(path,device):
            import numpy as np,torch
            data,meta=read(path,device)
            with np.load(spec['support_replay'],allow_pickle=False)as derived,np.load(path,allow_pickle=False)as original:
                if str(derived['parent_replay_sha256'])!=spec['replay_sha256']:raise ValueError('Support derivative parent differs')
                if any(not np.array_equal(original[k],derived[k])for k in original.files if k!='purpose'):raise ValueError('Support derivative altered original training fields')
                data['support']=torch.tensor(derived['support'],device=device,dtype=torch.float32)
            return data,meta
        print(json.dumps(run(a.output,spec,reader=reader,evaluator=evaluate,protocol='WM02_NEW_DECISION_DATA_V5',evaluate_dev=True)))
    return 0
if __name__=='__main__':raise SystemExit(main())
