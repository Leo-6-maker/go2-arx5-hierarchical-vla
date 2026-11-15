"""Same FP32 raw-plan codec, explicitly pinned to either frozen PD4 statistic."""
import json,sys
from pathlib import Path
import numpy as np
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.world_model.pd4_prospective_v71 import STATS
from scripts.pi0294.e1.artifacts import atomic_npz


def run(stats_path,pin):
    if pin not in STATS.values()or sha(stats_path)!=pin:raise ValueError('EXACT_FROZEN_PD4_STATISTICS_ONLY')
    stats=json.loads(Path(stats_path).read_bytes());q=Quantiles(stats['actions']['q01'],stats['actions']['q99'])
    for line in sys.stdin:
        r=json.loads(line);out=Path(r['output']);out.mkdir(parents=True,exist_ok=False)
        codec=GripLogitCodec(r['mount']);state=np.asarray(r['state10']);_,latent=codec.encode(state,r['policy_H40'])
        normalized=pad32(q.normalize(latent)).astype(np.float32)[None]
        container=out/'MODEL_CONTAINER.npz';atomic_npz(container,normalized=normalized)
        with np.load(container,allow_pickle=False)as z:actual=z['normalized'].copy()
        if not np.array_equal(actual,normalized):raise ValueError('REAL_SERIALIZED_FP32_CONTAINER')
        physical=codec.decode(state,pad32(q.unnormalize(actual[0,:,:7])))
        action=out/'ACTION.npz';atomic_npz(action,normalized=actual,physical_H40=physical)
        print(json.dumps(dict(passes=True,action=dict(path=str(action),sha256=sha(action)),model_loads=0,optimizer_updates=0,statistics_sha256=pin)),flush=True)


if __name__=='__main__':run(*sys.argv[1:])
