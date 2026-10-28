"""Original official baseline plus explicitly isolated, non-production causal arrays."""
import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10
from scripts.pi0294.data_cov1.codec import Trace
from scripts.pi0294.pv1.export import GATES, codec_packet, physical_errors


def active(processor):
    return processor.state_action_processor.norm_params['new_embodiment']['action']['eef_9d']


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


def relative(sap, state, target):
    from gr00t.data.types import ActionType, ActionFormat
    return sap._convert_to_relative_action(target[:,:9], state[:9], ActionType.EEF, ActionFormat.XYZ_ROT6D)


def diagnostic_inverse(packed, parameters, clamp):
    """Not a processor patch. Kept separate from official production decode."""
    values = np.clip(packed, -1., 1.) if clamp else packed.copy()
    a, b = parameters['min'], parameters['max']
    return (values+1.)/2.*(b-a)+a


def conditioning(value):
    x, y = value[:,3:6], value[:,6:9]
    rotation = matrix_from10(value)[:,:3,:3]
    return dict(row1_min_norm=float(np.linalg.norm(x,axis=-1).min()),
        row2_min_norm=float(np.linalg.norm(y,axis=-1).min()),
        min_cross_row_norm=float(np.linalg.norm(np.cross(x,y),axis=-1).min()),
        max_orthogonality_residual=float(np.abs(rotation@rotation.swapaxes(-1,-2)-np.eye(3)).max()),
        minimum_determinant=float(np.linalg.det(rotation).min()),
        rot6d='FIRST_TWO_MATRIX_ROWS;PROJECT_MATRIX_FLOAT64_GRAM_SCHMIDT')


def diagnostic_window(data, processor, destination, prefix):
    from gr00t.data.types import ActionType, ActionFormat
    sap = processor.state_action_processor
    state, target = data['state'], data['target']
    parameters = sap.norm_params['new_embodiment']['action']
    arrays = {k:np.asarray(v) for k,v in data.items()}
    summaries = []
    for dtype in (np.float32, np.float64):
        for clamp in (True, False):
            key = np.dtype(dtype).name+('_CLAMP' if clamp else '_NO_CLAMP_DIAGNOSTIC_ONLY')
            packed = data['normalized'].astype(dtype)
            decoded_relative = diagnostic_inverse(packed[:,:9],parameters['eef_9d'],clamp)
            grip = diagnostic_inverse(packed[:,9:],parameters['gripper_position'],clamp)
            absolute = sap._convert_to_absolute_action(decoded_relative,state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)
            actual = np.concatenate((absolute,grip),axis=-1)
            errors = physical_errors(target,actual)
            lr, rr = matrix_from10(target)[:,:3,:3], matrix_from10(actual)[:,:3,:3]
            rotation = lr@rr.swapaxes(-1,-2)
            skew = np.stack((rotation[:,2,1]-rotation[:,1,2], rotation[:,0,2]-rotation[:,2,0],
                             rotation[:,1,0]-rotation[:,0,1]),axis=-1)/2
            alternative = np.arctan2(np.linalg.norm(skew,axis=-1),(np.trace(rotation,axis1=-2,axis2=-1)-1)/2)
            delta = float(np.abs(actual-data['actual']).max())
            if dtype == np.float32 and clamp and delta > 1e-12:
                raise ValueError('CR1_DIAGNOSTIC_CLAMP32_MUST_REPRODUCE_ACTUAL_OFFICIAL_DECODE')
            arrays.update({key+'_transport':packed,key+'_inverse_clip_mask':np.abs(packed)>1,
                key+'_unnormalized_relative':decoded_relative,key+'_absolute':actual})
            summaries.append(dict(transport=np.dtype(dtype).name,original_inverse_clamp=clamp,
                diagnostic_only=True,errors=errors,passes_original_XYZ_SO3_gripper_gates=all(errors[k]<=GATES[k] for k in errors),
                alternative_SO3_atan2_max_rad=float(alternative.max()),alternative_changes_acceptance=False,
                official_decode_absolute_difference_max=delta,relative_conditioning=conditioning(
                    np.concatenate((decoded_relative,grip),axis=-1)),absolute_conditioning=conditioning(actual)))
    params = active(processor)
    arrays.update(active_min=params['min'],active_max=params['max'],active_isclose_mask=np.isclose(params['min'],params['max']))
    path = destination/(prefix+'.npz');np.savez_compressed(path,**arrays)
    from scripts.pi0294.fd1 import sha
    return dict(array_file=path.name,array_sha256=sha(path),anchor=int(prefix.rsplit('-',1)[1]),
        normalized_dtype=str(data['normalized'].dtype),state_dtype=str(state.dtype),target_dtype=str(target.dtype),
        packed_dtype=str(data['packed'].dtype),metric_matrix_dtype=str(matrix_from10(target).dtype),
        near_constant_active_channels=np.flatnonzero(np.isclose(params['min'],params['max'])).tolist(),
        target_conditioning=conditioning(target),cases=summaries)


class Observer(Trace):
    def __init__(self, processor, diagnostic_anchors=(), destination=None, prefix='', guard=lambda:None):
        super().__init__()
        self.processor,self.anchors,self.destination,self.prefix = processor,set(diagnostic_anchors),destination,prefix
        self.errors={k:[] for k in GATES}
        self.normalized_min=np.full(10,np.inf);self.normalized_max=np.full(10,-np.inf)
        self.outside_max=np.zeros(10);self.boundary_excess=np.zeros(9)
        self.failed_reference_indices=set();self.diagnostics=[];self.guard=guard

    def __call__(self, **data):
        self.guard();super().__call__(**data)
        target,actual,native,expected = (data[k] for k in ('target','actual','native_relative','expected_relative'))
        lr,rr=matrix_from10(target),matrix_from10(actual)
        cosine=np.clip((np.trace(lr[:,:3,:3]@rr[:,:3,:3].swapaxes(-1,-2),axis1=-2,axis2=-1)-1)/2,-1,1)
        errors=dict(XYZ_m=np.abs(target[:,:3]-actual[:,:3]).max(axis=-1),SO3_rad=np.arccos(cosine),
                    gripper=np.abs(target[:,9]-actual[:,9]),relative_SE3_max_abs=np.abs(native-expected[:,:9]).max(axis=-1))
        bad=np.zeros(40,dtype=bool)
        for k,values in errors.items():
            self.errors[k].append(values);bad |= values>GATES[k]
        self.failed_reference_indices.update((data['anchor']+np.flatnonzero(bad)).tolist())
        normalized=data['normalized'];self.normalized_min=np.minimum(self.normalized_min,normalized.min(axis=0))
        self.normalized_max=np.maximum(self.normalized_max,normalized.max(axis=0))
        self.outside_max=np.maximum(self.outside_max,np.maximum(np.abs(normalized)-1,0).max(axis=0))
        p=active(self.processor)
        self.boundary_excess=np.maximum(self.boundary_excess,np.maximum(np.maximum(p['min']-native,native-p['max']),0).max(axis=0))
        if data['anchor'] in self.anchors:
            self.diagnostics.append(diagnostic_window(data,self.processor,self.destination,self.prefix+'-'+str(data['anchor'])))

    def report(self):
        return dict(first_error=self.first_error,first_inverse_clip=self.first_inverse_clip,
            error_quantiles={k:dict(zip(('p0','p50','p95','p99','p100'),np.quantile(v,[0,.5,.95,.99,1]).tolist()))
                             for k,v in self.errors.items()},
            normalized_min=self.normalized_min.tolist(),normalized_max=self.normalized_max.tolist(),
            maximum_normalized_outside_amplitude_by_channel=self.outside_max.tolist(),
            maximum_physical_relative_boundary_excess_by_channel=self.boundary_excess.tolist(),
            failed_reference_indices=sorted(self.failed_reference_indices),diagnostics=self.diagnostics)


def scan(arrays, processor, guard, **kwargs):
    observer=Observer(processor,guard=guard,**kwargs)
    result=codec_packet(arrays,processor,guard,diagnostic_sink=observer)
    return dict(result,observation=observer.report())


def reproduction_matches(row):
    first=row['observation']['first_error']
    return (row['H40_windows_actually_checked']==601 and first is not None
            and first['window_anchor']==209 and first['action_index']==31 and first['reference_time_s']==12.
            and abs(first['error_at_action']['SO3_rad']-.00013817330386000284)<=1e-12
            and first['failed_physical_gates']==['SO3_rad'] and row['frozen_numeric_gates']==GATES)
