"""Numeric replay contract for new program responses; never old C10 rows."""
from collections import Counter
import numpy as np
from .fusion_live_program_replay_v1 import FEATURE_SHAPES

SCHEMA='WM02_LIVE_PROGRAM_RESPONSE_ARRAYS_V1'
ARMS=('RAW','SCALED','NO_FUTURE_ACTION')
OUTPUT_SHAPE=(16,8)
FEATURE_NAMES=tuple(sorted(FEATURE_SHAPES))
FEATURE_DIM=sum(int(np.prod(FEATURE_SHAPES[k]))for k in FEATURE_NAMES)
FEATURE_SLICES={}
offset=0
for name in FEATURE_NAMES:
    count=int(np.prod(FEATURE_SHAPES[name]));FEATURE_SLICES[name]=slice(offset,offset+count);offset+=count
FUTURE_SLICE=FEATURE_SLICES['planned_published_H40']


def validate(data):
    n=len(data['X'])
    if (data['X'].shape!=(n,FEATURE_DIM)or data['Y'].shape!=(n,*OUTPUT_SHAPE)
            or data['valid'].shape!=(n,16)or data['valid'].dtype.kind!='b'
            or not np.isfinite(data['X']).all()or not np.isfinite(data['Y']).all()
            or np.any(np.diff(data['valid'].astype(int),axis=1)>0)):
        raise ValueError('FINITE_PROGRAM_FEATURES_AND_OBSERVED_PREFIX_MASKS_REQUIRED')
    for key in ('split','parent','run_id','program_atom'):
        if data[key].shape!=(n,):raise ValueError('ONE_PROVENANCE_ROW_PER_ANCHOR_REQUIRED')
    if 'record_phase'in data and (data['record_phase'].shape!=(n,)or not set(data['record_phase']).issubset(
            {'PROGRAM','POST_PROGRAM_HOLD','POST_QUERY_HOLD_CONTROL'})):
        raise ValueError('DECLARED_PROGRAM_PHASE_IS_METADATA_ONLY')
    parents={};runs={}
    for p,r,s in zip(data['parent'],data['run_id'],data['split']):
        if s not in ('TRAIN','DEV','TEST','DIAGNOSTIC'):raise ValueError('EXPLICIT_ORIGINAL_DATA_ROLE_REQUIRED')
        if p in parents and parents[p]!=s:raise ValueError('PHYSICAL_PARENT_CANNOT_CROSS_SPLITS')
        if r in runs and runs[r]!=(p,s):raise ValueError('ONE_ORIGINAL_ASSIGNMENT_PER_PHYSICAL_RUN')
        parents[p]=s;runs[r]=(p,s)
    return {role:np.flatnonzero((data['split']==role)&data['valid'].any(1))for role in ('TRAIN','DEV','TEST','DIAGNOSTIC')}


def normalizers(data,train):
    if not len(train)or np.any(data['split'][train]!='TRAIN'):raise ValueError('ONLY_REAL_TRAIN_NORMALIZATION')
    x=data['X'][train].astype(np.float64);y=data['Y'][train].astype(np.float64);m=data['valid'][train,:,None]
    count=m.sum(0);den=np.maximum(count,1)
    mean=(y*m).sum(0)/den;variance=(((y-mean)**2)*m).sum(0)/den
    floor=np.array([1e-4]*3+[1e-3]*3+[1e-4]*2)
    return dict(x_mean=x.mean(0).astype('float32'),x_scale=np.maximum(x.std(0),1e-5).astype('float32'),
        y_mean=mean.astype('float32'),y_scale=np.maximum(np.sqrt(variance),floor).astype('float32'),
        supported_horizons=(count[:,0]>0))


def baseline_predictions(data,stats):
    n=len(data['X']);h=np.arange(1,17,dtype=float)[None,:,None]
    history=data['X'][:,FEATURE_SLICES['measured_EEF_history']].reshape(n,9,8)
    velocity=np.concatenate((-history[:,-2,:6],history[:,-1,6:]-history[:,-2,6:]),axis=1)
    return dict(ZERO_DISPLACEMENT=np.zeros((n,16,8)),PAST_CONSTANT_VELOCITY=h*velocity[:,None,:],
                TRAIN_MEAN_RESPONSE=np.broadcast_to(stats['y_mean'],(n,16,8)).copy())


def metrics(prediction,data,ids):
    result={}
    for h in (1,4,8,16):
        selected=ids[data['valid'][ids,h-1]];error=prediction[selected,h-1]-data['Y'][selected,h-1]
        result[str(h)]=dict(anchors=len(selected),physical_runs=len(set(data['run_id'][selected])),
            XYZ_vector_RMSE_mm=(float(np.sqrt(np.mean(np.sum(error[:,:3]**2,axis=1))))*1000 if len(selected)else None),
            rotation_vector_RMSE_rad=(float(np.sqrt(np.mean(np.sum(error[:,3:6]**2,axis=1))))if len(selected)else None),
            finger_coordinate_RMSE_mm=(float(np.sqrt(np.mean(error[:,6:]**2)))*1000 if len(selected)else None))
    return result


def summarize(data):
    return dict(anchors=len(data['X']),physical_runs=len(set(data['run_id'])),
        roles=dict(Counter(data['split'].tolist())),observed_label_atoms=int(data['valid'].sum()),
        features=FEATURE_DIM,output=list(OUTPUT_SHAPE),anchors_are_independent_parents=False,
        declared_program_phases=(dict(Counter(data['record_phase'].tolist()))if 'record_phase'in data else None))


def metrics_by_phase(prediction,data,ids):
    out=dict(ALL=metrics(prediction,data,ids))
    for phase in ('PROGRAM','POST_PROGRAM_HOLD','POST_QUERY_HOLD_CONTROL'):
        if 'record_phase'in data:
            out[phase]=metrics(prediction,data,ids[data['record_phase'][ids]==phase])
    return out
