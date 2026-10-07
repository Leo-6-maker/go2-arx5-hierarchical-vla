"""Qualified ordered cache, with hash-bound paths for multiple forks of one seed."""
import importlib.util,sys
from functools import lru_cache
from pathlib import Path
SOURCE=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run/code/cpu_inputs.py')
spec=importlib.util.spec_from_file_location('pi0279_qualified_inputs',SOURCE)
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
read,write,sha,plain,ItemLedger,STATS_SHA=q.read,q.write,q.sha,q.plain,q.ItemLedger,q.STATS_SHA


class Inputs(q.Inputs):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.paths={}
        for r in self.rows.values():
            key=(r['seed'],r['command_sha256'],r['observation_sha256']);folder=Path(r['episode'])
            assert key not in self.paths or self.paths[key]==folder
            self.paths[key]=folder

    @lru_cache(maxsize=28)
    def episode(self,seed,command_hash,observation_hash):
        import numpy as np
        folder=self.paths[seed,command_hash,observation_hash]
        assert sha(folder/'commands.npz')==command_hash and sha(folder/'observations.npz')==observation_hash
        with np.load(folder/'commands.npz') as z:commands=z['action'].copy()
        with np.load(folder/'observations.npz') as z:obs={k:z[k].copy() for k in ('state_v1','base_camera_96','hand_camera_96')}
        return commands,obs


def loader(*a,**kw):return q.loader(*a,**kw)
