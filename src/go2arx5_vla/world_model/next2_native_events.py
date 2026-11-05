"""Contact outcomes over all25 actual native ticks, including brief contacts."""
import numpy as np

def contact_outcomes(rows,pad_ids,cube):
    any_side=np.zeros(2,dtype=bool);peak=np.zeros(2);opposing=False;positive_ticks=0
    for row in rows:
        sides=[[],[]]
        for c in row['contacts']:
            for i,pad in enumerate(pad_ids):
                if {c['geom0'],c['geom1']}=={pad,cube}:
                    force=float(c['wrench'][0]);peak[i]=max(peak[i],force)
                    if force>=.05:sides[i].append(c['normal']if c['geom1']==cube else-np.asarray(c['normal']))
        observed=np.asarray([bool(s)for s in sides]);any_side|=observed;positive_ticks+=bool(observed.any());opposing|=bool(sides[0]and sides[1]and any(np.dot(a,b)<=-.5 for a in sides[0]for b in sides[1]))
    return any_side,peak,opposing,positive_ticks
