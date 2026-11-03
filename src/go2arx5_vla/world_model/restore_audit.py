"""Recursive comparison of complete native/controller snapshots, no physics."""
import numpy as np


def compare(left,right,*,tolerance=1e-10):
    maximum=0.;mismatches=[]
    def walk(a,b,path):
        nonlocal maximum
        if isinstance(a,dict):
            if not isinstance(b,dict)or a.keys()!=b.keys():mismatches.append(path+':keys');return
            for k in a:walk(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(tuple,list)):
            if not isinstance(b,(tuple,list))or len(a)!=len(b):mismatches.append(path+':length');return
            for i,(x,y)in enumerate(zip(a,b)):walk(x,y,path+'.'+str(i))
        elif isinstance(a,np.ndarray):
            if np.shape(a)!=np.shape(b):mismatches.append(path+':shape');return
            if np.issubdtype(a.dtype,np.number):
                delta=float(np.max(np.abs(a-np.asarray(b))))if a.size else 0.
                maximum=max(maximum,delta)
                if not np.isfinite(delta)or delta>tolerance:mismatches.append(path)
            elif not np.array_equal(a,b):mismatches.append(path)
        elif a!=b:mismatches.append(path)
    walk(left,right,'restore')
    return dict(max_abs_error=maximum,mismatches=mismatches,tolerance=tolerance,passed=not mismatches)
