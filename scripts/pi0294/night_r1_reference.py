"""Quantized local timing and installed/executed reference history contracts.

These are task-side CPU tools. A saved-input result never grants native startup
or full-path collision qualification, and actual qpos is never command history.
"""
import hashlib
import numpy as np

FACTORS=(1.,1.25,1.5,2.)


def joint_rates(q,t):
    q,t=np.asarray(q,dtype=float),np.asarray(t,dtype=float)
    if q.ndim!=2 or q.shape[0]!=len(t) or len(t)<3 or not np.isfinite(q).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('FINITE_ORDERED_REFERENCE_REQUIRED')
    dt=np.diff(t);v=np.diff(q,axis=0)/dt[:,None]
    a=2*np.diff(v,axis=0)/(dt[1:]+dt[:-1])[:,None]
    return v,a


def stencils(q,t,*,connector_end=None,source_boundaries=(),chunk_boundaries=()):
    v,a=joint_rates(q,t);t=np.asarray(t);rows=[]
    for i,j in np.argwhere(np.abs(a)>1.*(1+1e-10)):
        lo,center,hi=map(float,t[i:i+3]);labels=[]
        if connector_end is not None and lo<=connector_end<=hi:labels.append('CONNECTOR_TO_SKILL')
        if any(lo<=x<=hi for x in source_boundaries):labels.append('SOURCE_KNOT_OR_BLOCK')
        if any(lo<=x<=hi for x in chunk_boundaries):labels.append('H40_SUBMISSION_BOUNDARY')
        if not labels:labels.append('WITHIN_SEGMENT')
        rows.append(dict(index=int(i),joint=int(j),time_s=[lo,center,hi],acceleration=float(a[i,j]),
                         labels=labels,connector_center_distance_s=None if connector_end is None else center-connector_end))
    return dict(max_velocity=float(np.abs(v).max()),max_acceleration=float(np.abs(a).max()),violations=rows)


def local_quantized_candidates(dense_q,dense_time,*,grid=.05,max_duration=128.):
    """A bounded rate-constraint solve followed by exactly four fixed factors.

    Keep all original path knots. The local solver only lengthens intervals
    implicated by rate stencils; each final 20Hz resampling is rechecked.
    Return every rejected candidate rather than silently smoothing its values.
    """
    q,t=np.asarray(dense_q,dtype=float),np.asarray(dense_time,dtype=float)
    if (q.ndim!=2 or len(t)!=len(q) or len(t)<3 or not np.isfinite(t).all()
        or np.any(np.diff(t)<=0) or not np.isfinite(q).all()
        or grid!=.05 or max_duration not in (32.,64.,96.,128.)):
        raise ValueError('DENSE_REFERENCE_IDENTITY')
    old=np.r_[np.arange(t[0],t[-1]-1e-10,grid),t[-1]]
    knots=np.stack([np.interp(old,t,q[:,j]) for j in range(q.shape[1])],axis=1)
    original_dt=np.diff(old);dt=original_dt.copy();dq=np.diff(knots,axis=0)
    dt=np.maximum(dt,np.max(np.abs(dq),axis=1)/.7)
    converged=False
    for iteration in range(24):
        v=dq/dt[:,None];a=2*np.diff(v,axis=0)/(dt[1:]+dt[:-1])[:,None]
        ratios=np.max(np.abs(a),axis=1);bad=np.flatnonzero(ratios>1.*(1+1e-10))
        if not len(bad):converged=True;break
        multiplier=np.ones_like(dt)
        for i in bad:multiplier[i:i+2]=np.maximum(multiplier[i:i+2],np.sqrt(ratios[i])*(1+1e-6))
        dt*=multiplier
    changed=dt>original_dt+1e-12;attempts=[];selected=None
    for factor in FACTORS:
        candidate=np.where(changed,dt*factor,original_dt)
        candidate=np.ceil(candidate/grid-1e-10)*grid
        new=np.r_[old[0],old[0]+np.cumsum(candidate)]
        if new[-1]-new[0]>max_duration:
            attempts.append(dict(factor=factor,status='EXCEEDS_PREDECLARED_MAX_DURATION',duration_s=float(new[-1]-new[0])));continue
        uniform=np.arange(round((new[-1]-new[0])/grid)+1)*grid+new[0]
        original_parameter=np.interp(uniform,new,old)
        resampled=np.stack([np.interp(original_parameter,t,q[:,j]) for j in range(q.shape[1])],axis=1)
        rates=stencils(resampled,uniform)
        passed=rates['max_velocity']<=.7*(1+1e-10) and rates['max_acceleration']<=1.*(1+1e-10)
        proof=dict(factor=factor,status='PASS_SAVED_JOINT_WITNESS_ONLY' if passed else 'RATE_REJECT_AFTER_QUANTIZATION',
            duration_s=float(new[-1]-new[0]),max_velocity=rates['max_velocity'],max_acceleration=rates['max_acceleration'],
            changed_intervals=int(changed.sum()),continuous_source_path_unchanged=True,
            quantized_samples_are_not_all_dense_knots=True,physics_qualification=False)
        attempts.append(proof)
        if passed and selected is None:selected=dict(knot_times=new,original_knot_times=old,uniform_times=uniform,original_parameter=original_parameter,joint_witness=resampled,proof=proof)
    return dict(selected=selected,attempts=attempts,solver_converged=converged,solver_iterations=iteration+1,
                input_q_sha256=hashlib.sha256(q.astype('<f8').tobytes()).hexdigest(),input_time_sha256=hashlib.sha256(t.astype('<f8').tobytes()).hexdigest())


class ExecutedReferenceHistory:
    """Keep installed plans separate from the references actually consumed."""
    def __init__(self):self.pending={};self.installed={};self.executed=[]

    def propose(self,identity,poses,joint_witness,source_times,*,native_start_time,witness_frame_id):
        if identity in self.pending or identity in self.installed:raise ValueError('DUPLICATE_REFERENCE_ID')
        poses,q=np.asarray(poses).copy(),np.asarray(joint_witness).copy()
        if poses.shape!=(40,4,4) or q.shape!=(40,6) or not np.isfinite(poses).all() or not np.isfinite(q).all():raise ValueError('REFERENCE_SHAPE')
        source=np.asarray(source_times,dtype=float)
        if (source.shape!=(40,) or not np.isfinite(source).all() or np.any(np.diff(source)<0)
            or not np.isfinite(native_start_time) or not isinstance(witness_frame_id,str) or not witness_frame_id):
            raise ValueError('SOURCE_AND_NATIVE_CLOCKS_REQUIRED')
        self.pending[identity]=dict(poses=poses,q=q,source_times=source.copy(),native_start=float(native_start_time),witness_frame_id=witness_frame_id)

    def reject(self,identity):self.pending.pop(identity)

    def acknowledge_install(self,identity,ack_identity):
        if identity!=ack_identity:raise ValueError('INSTALL_ACK_IDENTITY')
        self.installed[identity]=self.pending.pop(identity)

    def acknowledge_executed(self,identity,prefix_index,native_step,actual_end_time):
        if identity not in self.installed or not 0<=prefix_index<8:raise ValueError('EXECUTED_REFERENCE_NOT_INSTALLED_PREFIX')
        if type(prefix_index) is not int or type(native_step) is not int:raise ValueError('INTEGER_PREFIX_AND_NATIVE_STEP')
        p=self.installed[identity];source=float(p['source_times'][prefix_index]);native_start=p['native_start']+prefix_index*.05
        if self.executed and (native_step<=self.executed[-1]['native_step'] or native_start<=self.executed[-1]['native_start_time']+1e-10
                             or source<self.executed[-1]['source_time']-1e-10):raise ValueError('REFERENCE_REPLAY_OR_NONMONOTONE')
        if abs(float(actual_end_time)-native_start-.05)>1e-8:raise ValueError('COMMAND_NATIVE_END_TIME_BINDING')
        self.executed.append(dict(identity=identity,prefix_index=prefix_index,native_step=int(native_step),source_time=source,
                                  native_start_time=native_start,actual_end_time=float(actual_end_time),pose=p['poses'][prefix_index].copy(),joint_witness=p['q'][prefix_index].copy(),
                                  witness_frame_id=p['witness_frame_id'],joint_witness_is_commanded_joint_action=False))

    def last_two(self):
        if len(self.executed)<2:raise ValueError('TWO_REAL_EXECUTED_REFERENCE_SAMPLES_REQUIRED')
        return self.executed[-2:]

    def check_next(self,joint_witness,*,native_start_time,witness_frame_id):
        past=self.last_two();q=np.asarray(joint_witness,dtype=float)
        if any(r['witness_frame_id']!=witness_frame_id for r in past):
            raise ValueError('COMMON_ROOT_AND_IK_BRANCH_WITNESS_REQUIRED')
        if q.shape!=(40,6) or abs(native_start_time-past[-1]['actual_end_time'])>1e-8:
            raise ValueError('CONTIGUOUS_NEXT_REFERENCE_REQUIRED')
        times=np.r_[[r['native_start_time'] for r in past],native_start_time+np.arange(40)*.05]
        value=stencils(np.concatenate([np.stack([r['joint_witness'] for r in past]),q]),times,
                       chunk_boundaries=(native_start_time,))
        value['uses_actual_qpos_as_reference']=False
        return value
