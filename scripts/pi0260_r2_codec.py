"""Actual GR00T SE(3) codec with the protocol's unclamped q01/q99 affine map."""
import copy
import numpy as np

TAG='new_embodiment'
IDENTITY='PI0260_R2_C_KIN_MOTION2_V1'

def bounds(params):
    lo=np.asarray(params.get('q01',params['min']))
    hi=np.asarray(params.get('q99',params['max']))
    return lo,np.maximum(hi-lo,1e-6)

def normalize(values,params):
    lo,width=bounds(params);return 2*(values-lo)/width-1

def unnormalize(values,params):
    lo,width=bounds(params);return (values+1)*width/2+lo

def install():
    # The pinned stock inverse clips unconditionally; R2 explicitly requires no clamp.
    # Keep its actual pose/chunk conversion, replacing only the declared affine functions.
    from gr00t.data.state_action import state_action_processor as module
    module.normalize_values_minmax=normalize
    module.unnormalize_values_minmax=unnormalize
    return module.StateActionProcessor

def modality():
    from gr00t.data.types import ModalityConfig,ActionConfig,ActionFormat,ActionType,ActionRepresentation
    return {TAG:dict(
        video=ModalityConfig(delta_indices=[0],modality_keys=['external_rgb','wrist_rgb']),
        state=ModalityConfig(delta_indices=[0],modality_keys=['eef_9d','gripper_position']),
        action=ModalityConfig(delta_indices=list(range(40)),modality_keys=['eef_9d','gripper_position'],action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE,type=ActionType.EEF,format=ActionFormat.XYZ_ROT6D,state_key='eef_9d'),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT,state_key='gripper_position')]),
        language=ModalityConfig(delta_indices=[0],modality_keys=['annotation.language.task_description']))}

def processor(statistics=None):
    cls=install()
    return cls(modality(),statistics=statistics,use_percentiles=True,clip_outliers=False,
               apply_sincos_state_encoding=False,use_relative_action=True)

def relative(raw,state):
    from gr00t.data.types import ActionFormat,ActionType
    return processor()._convert_to_relative_action(raw[:,:9],state[:9],ActionType.EEF,ActionFormat.XYZ_ROT6D)

def stats(values,axis):
    values=np.asarray(values,dtype=np.float64)
    return {k:np.asarray(v).tolist() for k,v in dict(min=values.min(axis),max=values.max(axis),mean=values.mean(axis),
        std=values.std(axis),q01=np.quantile(values,.01,axis=axis,method='linear'),q99=np.quantile(values,.99,axis=axis,method='linear')).items()}

def fit(states,actions,relatives):
    # Exactly one fit on the frozen selected TRAIN population, before codec exclusion.
    states=np.asarray(states);actions=np.asarray(actions);relatives=np.asarray(relatives)
    assert states.ndim==2 and states.shape[1]==10 and actions.shape[1:]==(40,10) and relatives.shape[1:]==(40,9)
    return {TAG:dict(state=dict(eef_9d=stats(states[:,:9],0),gripper_position=stats(states[:,9:],0)),
        action=dict(eef_9d=stats(actions[:,:,:9],(0,1)),gripper_position=stats(actions[:,:,9:],(0,1))),
        relative_action=dict(eef_9d=stats(relatives,0)))}

def encode(proc,state,action=None):
    s=dict(eef_9d=state[None,:9],gripper_position=state[None,9:])
    a={} if action is None else dict(eef_9d=action[:,:9],gripper_position=action[:,9:])
    if action is None:proc.eval()
    else:proc.train()
    ps,pa=proc.apply(s,a,TAG)
    packed_state=np.pad(np.concatenate([ps[k] for k in ('eef_9d','gripper_position')],-1),((0,0),(0,122))).astype(np.float32)
    if action is None:return packed_state
    packed=np.pad(np.concatenate([pa[k] for k in ('eef_9d','gripper_position')],-1),((0,0),(0,122))).astype(np.float32)
    mask=np.zeros((40,132),np.float32);mask[:,:10]=1
    return packed_state,packed,mask

def decode(proc,packed,state):
    a=proc.unapply_action(dict(eef_9d=packed[:,:9],gripper_position=packed[:,9:10]),TAG,
        state=dict(eef_9d=state[None,:9],gripper_position=state[None,9:]))
    return np.concatenate([a['eef_9d'],a['gripper_position']],-1)

def check():
    p=dict(min=np.array([0.,2.]),max=np.array([1.,2.]),q01=np.array([0.,2.]),q99=np.array([1.,2.]))
    x=np.array([[-.2,2.],[1.3,2.000003]],np.float32)
    assert np.allclose(unnormalize(normalize(x,p).astype(np.float32),p),x,atol=1e-7)
    assert normalize(x,p)[0,0]<-1 and normalize(x,p)[1,0]>1
    return dict(status='PASS_AFFINE_NO_CLAMP_FLOOR',identity=IDENTITY)

if __name__=='__main__':print(check())
