"""Compare the observed first backward pass with the original serial computation."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.sla.hopper_parent_custody import digest, write_once

COMPONENTS = ('target','robodojo','libero','aux','total')


class FirstStep:
    def __init__(self, root, reference=None):
        self.root=Path(root)
        self.root.mkdir()
        self.reference=Path(reference) if reference else None
        self.expected=json.loads((self.reference/'COMPLETE.json').read_bytes()) if reference else None
        self.inputs={d:[] for d in COMPONENTS[:-1]}
        self.results={}

    def batch(self, component, args):
        import jax

        # Hash precisely the arrays handed to the actual backward kernel.
        leaves=jax.tree_util.tree_flatten_with_path(args)[0]
        host=[(jax.tree_util.keystr(k),np.asarray(a)) for k,a in leaves]
        length=len(host[0][1])
        if any(len(a)!=length for _,a in host):
            raise ValueError('Inconsistent example axis in actual kernel inputs')
        for i in range(length):
            h=hashlib.sha256()
            for key,array in host:
                part=np.ascontiguousarray(array[i])
                h.update(json.dumps([key,str(part.dtype),part.shape]).encode())
                h.update(part.tobytes())
            self.inputs[component].append(h.hexdigest())

    def component(self, name, loss, tree):
        import jax

        if name in self.results or name not in COMPONENTS:
            raise ValueError('Unexpected repeated gradient component')
        if loss is not None and not np.isfinite(loss):
            raise ValueError('Nonfinite component loss')
        prior=self.expected['components'][name] if self.expected else None
        err=0. if loss is None or prior is None else abs(loss-prior['loss'])/max(abs(prior['loss']),1e-12)
        ref2=0.
        diff2=0.
        leaves=[]
        for i,(key,grad) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            actual=np.asarray(grad)
            file=f'{name}-{i:03d}.npy'
            row=dict(key=jax.tree_util.keystr(key),shape=list(actual.shape),dtype=str(actual.dtype),file=file)
            if actual.dtype!=np.float32 or not np.isfinite(actual).all():
                raise ValueError('Expected finite FP32 gradient')
            if prior is None:
                np.save(self.root/file,actual,allow_pickle=False)
                row['sha256']=digest(self.root/file)
            else:
                expected=prior['leaves'][i]
                path=self.reference/file
                if any(row[k]!=expected[k] for k in row) or digest(path)!=expected['sha256']:
                    raise ValueError('Reference gradient leaf identity differs')
                saved=np.load(path,allow_pickle=False,mmap_mode='r').reshape(-1)
                flat=actual.reshape(-1)
                for pos in range(0,len(flat),262144):
                    ref=np.array(saved[pos:pos+262144],dtype=np.float64)
                    delta=flat[pos:pos+262144].astype(np.float64)-ref
                    diff2+=float(np.dot(delta,delta))
                    ref2+=float(np.dot(ref,ref))
            leaves.append(row)
        if prior and len(leaves)!=len(prior['leaves']):
            raise ValueError('Gradient tree membership differs')
        rel=math.sqrt(diff2)/max(math.sqrt(ref2),1e-12)
        self.results[name]=dict(loss=loss,loss_relative=err,gradient_relative_L2=rel,leaves=leaves,
                                passed=math.isfinite(rel) and err<=1e-4 and rel<=1e-3)

    def finish(self):
        if set(self.results)!=set(COMPONENTS) or {d:len(v) for d,v in self.inputs.items()}!=dict(
                target=64,robodojo=32,libero=32,aux=4):
            raise ValueError('First-step component or input readback incomplete')
        same=self.expected is None or self.inputs==self.expected['kernel_inputs']
        proof=dict(components=self.results,kernel_inputs=self.inputs,inputs_equal=same,
                   passed=same and all(v['passed'] for v in self.results.values()))
        write_once(self.root/'COMPLETE.json',proof)
        return proof['passed']
