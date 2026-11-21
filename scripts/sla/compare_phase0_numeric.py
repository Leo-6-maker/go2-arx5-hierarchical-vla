"""Compare sealed per-window diagnostics. Output only hashes, shapes and numerical summaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hopper_numeric_probe import CHECKPOINT, ORDINALS, PACKAGE, PANEL, pin, sha


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError('Empty or nonfinite comparison')
    return dict(max=float(a.max()),p95=float(np.quantile(a,.95)),rms=float(np.sqrt(np.mean(a**2))))


def norm(a):
    return np.linalg.norm(a,axis=-1) if a.ndim>1 else np.abs(a)


def angle(a, b):
    for rotations in (a,b):
        if (rotations.shape!=(40,3,3) or not np.isfinite(rotations).all() or
                np.max(np.abs(rotations@np.swapaxes(rotations,-1,-2)-np.eye(3)))>2e-5 or
                np.any(np.abs(np.linalg.det(rotations)-1)>2e-5)):
            raise ValueError('Invalid SO(3) input')
    r = a @ np.swapaxes(b,-1,-2)
    sine = np.linalg.norm(np.stack([r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]],axis=-1),axis=-1)/2
    return np.arctan2(sine,(np.trace(r,axis1=1,axis2=2)-1)/2)


def evidence(folder, ordinal):
    stem=f'{ordinal:03d}'
    rec=json.loads((folder/(stem+'.json')).read_bytes())
    f=folder/(stem+'.private.npz')
    if (sha(f)!=rec['artifact_sha256'] or rec['ordinal']!=ordinal or
            not (folder/(stem+'.intent.json')).is_file()):
        raise ValueError('Wrong observation archive or missing intent')
    with np.load(f,allow_pickle=False) as z:
        a={k:np.asarray(z[k]) for k in z.files}
    required={'noise':(1,40,32),'normalized':(40,32),'physical':(40,10),'rotations':(40,3,3),
              'width_m':(40,),'previous_width_m':(),'linear_velocity':(39,3),'angular_velocity':(39,3),
              'linear_acceleration':(40,3),'angular_acceleration':(40,3),'finger_velocity':(39,)}
    for k,shape in required.items():
        if a[k].shape!=shape or not np.isfinite(a[k]).all():
            raise ValueError('Unexpected '+k)
    for name,h in [('normalized',rec['outputs']['normalized_sha256']),('physical',rec['outputs']['decoded_sha256']),
                   ('noise',rec['tensors']['noise']['sha256'])]:
        if hashlib.sha256(a[name].tobytes(order='C')).hexdigest()!=h:
            raise ValueError('Array and independently recorded original digest disagree')
    return rec,a


def compare_one(left, right, ordinal, bounds):
    ar,a=evidence(left,ordinal)
    br,b=evidence(right,ordinal)
    if any(ar[k]!=br[k] for k in ('index','input_sha256','noise_id')):
        raise ValueError('Original observation or seed differs')
    if set(ar['tensors'])!=set(br['tensors']):
        raise ValueError('Different sets of captured tensors')
    same={k:v==br['tensors'][k] for k,v in ar['tensors'].items()}
    data={k:np.asarray(a[k],dtype=np.float64)-np.asarray(b[k],dtype=np.float64)
          for k in ('linear_velocity','angular_velocity','finger_velocity','linear_acceleration','angular_acceleration')}
    distances={'translation_m':norm(a['physical'][:,:3].astype(float)-b['physical'][:,:3].astype(float)),
               'orientation_rad':angle(a['rotations'].astype(float),b['rotations'].astype(float)),
               'width_m':np.abs(a['width_m']-b['width_m'])}
    for key,units in [('linear_velocity','m_s'),('angular_velocity','rad_s'),('finger_velocity','m_s'),
                      ('linear_acceleration','m_s2'),('angular_acceleration','rad_s2')]:
        distances[key+'_'+units]=norm(data[key])
    seam1=(a['width_m'][0]-a['previous_width_m'])/.05
    seam2=(b['width_m'][0]-b['previous_width_m'])/.05
    if a['previous_width_m']!=b['previous_width_m'] or not all(np.isclose(abs(s),r['seam']['first_seam_speed_m_s'],rtol=1e-7,atol=1e-10) for s,r in [(seam1,ar),(seam2,br)]):
        raise ValueError('Recorded seam does not reproduce actual prior command')
    distances['seam_velocity_m_s']=np.array([abs(seam1-seam2)])
    results={}
    for k,v in distances.items():
        width=9 if 'velocity' in k and k!='seam_velocity_m_s' else 10
        results[k]=dict(H40=distribution(v),P10=distribution(v[:width]),within_bound=bool(v.max()<=bounds[k]))
    gates={}
    for name in ('nominal','reference'):
        g1=ar['envelope']['paths'][name]['channels']
        g2=br['envelope']['paths'][name]['channels']
        gates[name]={}
        for k,x in g1.items():
            y=g2[k]
            if x['threshold']!=y['threshold']:
                raise ValueError('Different native channel limits')
            gates[name][k]=dict(threshold=x['threshold'],left_peak_margin=x['margin'],right_peak_margin=y['margin'],
                left_first=x['first_exceedance_k'],right_first=y['first_exceedance_k'],
                same_exceeded_indices=x['exceedance_indices']==y['exceedance_indices'])
    trace={}
    for k in sorted(a.keys()|b.keys()):
        if k=='flow' or k=='prefix_tokens' or k.startswith('prefix_cache_'):
            if k not in b or k not in a or a[k].shape!=b[k].shape:
                raise ValueError('Incompatible instrumented arrays')
            diff=np.abs(a[k].astype(np.float64)-b[k].astype(np.float64))
            trace[k]=[distribution(step) for step in diff] if k=='flow' else distribution(diff)
    normalized=np.abs(a['normalized'].astype(float)-b['normalized'].astype(float))
    return dict(ordinal=ordinal,index=ar['index'],model_input_sha256=ar['input_sha256'],same_tensors=same,
                noise_error=distribution(np.abs(a['noise'].astype(float)-b['noise'].astype(float))),
                normalized_error=dict(active7=distribution(normalized[:,:7]),padding25=distribution(normalized[:,7:])),
                physical=results,guards_equal=ar['guard']==br['guard'],guards=[ar['guard'],br['guard']],
                original_channels=gates,trace=trace,own_v1=[ar['own_v1'],br['own_v1']])


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('left',type=Path)
    cli.add_argument('right',type=Path)
    cli.add_argument('preregistration',type=Path)
    cli.add_argument('spec_sha')
    cli.add_argument('mode',choices=['ordinary2','trace2','ordinary16'])
    args=cli.parse_args()
    spec=pin(args.preregistration,args.spec_sha)
    if (spec['checkpoint'],spec['package'],spec['panel'])!=(CHECKPOINT,PACKAGE,PANEL):
        raise ValueError('Unexpected numeric scope')
    ords=ORDINALS if args.mode=='ordinary16' else ORDINALS[:2]
    for p,site in [(args.left,'hopper'),(args.right,'core')]:
        cfg=json.loads((p/'CONFIG.json').read_bytes())
        if cfg['preregistration']!=args.spec_sha or cfg['site']!=site:
            raise ValueError('Unexpected site or preregistration')
        terminal=json.loads((p/args.mode/'RESULT.json').read_bytes())
        if terminal['completed']!=len(ords):
            raise ValueError('Missing measurements')
    print(json.dumps(dict(schema='SLA_PHASE0_NUMERIC_MEASUREMENTS_V2',mode=args.mode,
        checkpoint=CHECKPOINT,preregistration=args.spec_sha,native_admission=False,
        rows=[compare_one(args.left/args.mode,args.right/args.mode,i,
                          spec['equivalence']['per_point_absolute_limits']) for i in ords]),
        sort_keys=True,allow_nan=False))
