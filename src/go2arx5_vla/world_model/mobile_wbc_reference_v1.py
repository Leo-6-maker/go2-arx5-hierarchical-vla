"""New mobile-only reference boundary; no changes to the frozen WBC."""
import numpy as np
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0294.base_intent import eef_document
from ..trajectory.so3 import matrix_to_quat,rotation_matrix_to_rot6d_rows
from .fusion_trajectory_adapter import _geometry,WORKING_OPEN_M,MECHANICAL_TRAVEL_M
from .native_collection_performance_v1 import profiled

SCHEMA='MOBILE_WBC_SLA_BOUNDARY_V1'
PHASE='collect-target-mobile-wbc-boundary-v1'
DT=.05
START=5703
M0_END=8703
AXES=('+x','-x','+y','-y','+yaw','-yaw')


def quintic(t):
    t=np.clip(np.asarray(t,dtype=float),0.,1.)
    return t**3*(10.-15.*t+6.*t*t)


def eef_translation_fraction(seconds):
    # Six seconds, with 1.5s smooth velocity ramps and a 3s plateau.
    # Peak speed 0.05/4.5m/s also passes the unchanged H40 endpoint stop gate.
    t=np.clip(np.asarray(seconds,float),0.,6.)
    def integral(u):return 2.5*u**4-3.*u**5+u**6
    first=1.5*integral(t/1.5)
    middle=t-.75
    last=4.5-1.5*integral((6.-t)/1.5)
    return np.where(t<1.5,first,np.where(t>4.5,last,middle))/4.5


def m0_reference(tcp,mount,axis,eef_mode,elapsed):
    if axis not in AXES or eef_mode not in('WORLD_HOLD','SMOOTH_5CM_TRANSLATION'):
        raise ValueError('FIXED_TWELVE_M0_CONDITIONS')
    if type(elapsed)is not int or elapsed<0:raise ValueError('EXACT_50MS_MOBILE_CURSOR')
    tcp,mount=np.asarray(tcp,float),np.asarray(mount,float)
    if tcp.shape!=mount.shape or tcp.shape!=(4,4)or not np.isfinite([tcp,mount]).all():
        raise ValueError('FINITE_MEASURED_WORLD_TCP_AND_MOUNT')
    weights=quintic((elapsed+np.arange(41))*DT/4.)
    ee=np.repeat(tcp[None],41,axis=0);base=np.repeat(mount[None],41,axis=0)
    if eef_mode=='SMOOTH_5CM_TRANSLATION':
        ee[:,0,3]+=.05*eef_translation_fraction((elapsed+np.arange(41))*DT)
    sign=1. if axis[0]=='+'else-1.
    if axis[-1]=='x':base[:,0,3]+=sign*.1*weights
    elif axis[-1]=='y':base[:,1,3]+=sign*.1*weights
    else:
        for i,angle in enumerate(sign*np.deg2rad(10.)*weights):
            c,s=np.cos(angle),np.sin(angle)
            base[i,:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@mount[:3,:3]
    # Forty learning-clock points; the 41st is explicitly wire padding.
    action=np.empty((40,10));action[:,:3]=ee[:40,:3,3]
    action[:,3:9]=rotation_matrix_to_rot6d_rows(ee[:40,:3,:3]);action[:,9]=WORKING_OPEN_M/MECHANICAL_TRAVEL_M
    return action,base


def full_eef_gate(action,anchor):
    from . import fusion_trajectory_adapter as e
    a=np.asarray(action,float)
    if a.shape!=(40,10)or not np.isfinite(a).all()or np.any((a[:,9]<0)|(a[:,9]>1)):
        raise ValueError('MOBILE_FINITE_FULL40_AND_ORIGINAL_GRIP_RANGE')
    value=_geometry(a,np.asarray(anchor,float))
    limits=dict(maximum_translation_speed_m_s=e.MAX_TRANSLATION_SPEED_M_S,
        maximum_angular_speed_rad_s=e.MAX_ANGULAR_SPEED_RAD_S,
        maximum_finger_speed_m_s=e.MAX_FINGER_SPEED_M_S,
        maximum_discrete_translation_acceleration_m_s2=e.MAX_TRANSLATION_ACCELERATION_M_S2,
        maximum_discrete_angular_acceleration_rad_s2=e.MAX_ANGULAR_ACCELERATION_RAD_S2)
    if any(value[k]>v*(1+1e-10)for k,v in limits.items()):
        raise ValueError('MOBILE_ORIGINAL_FULL40_EEF_GATE')
    return value


def document(action,*,mode,base_poses=None):
    doc=eef_document(action)
    if mode=='M_FREE':
        if base_poses is not None:raise ValueError('M_FREE_MUST_OMIT_BASE_NOT_HOLD_IT')
        return doc
    if mode!='M_ASSIST':raise ValueError('TWO_EXPLICIT_MOBILE_MODES')
    base=np.asarray(base_poses,float)
    if base.shape!=(41,4,4)or not np.isfinite(base).all():raise ValueError('WORLD_BASE_41_WIRE_ENDPOINTS')
    # World poses bound once. No future measured base or 12D velocity input.
    doc['base']=dict(frame='world',start_delay=0.,lead_in=0.,waypoints=[
        dict(time=k*DT,position=p[:3,3].tolist(),quaternion_xyzw=matrix_to_quat(p[:3,:3]).tolist())
        for k,p in enumerate(base)])
    return doc


def validate_preview(adapter,action,anchor):
    program=getattr(adapter,'mobile_program',None)
    if adapter.spec.get('mobile_stage')!='L1'or program is None:
        return full_eef_gate(action,anchor)
    # A renewal is the same previously certified H40's suffix, not a fresh
    # rest-to-rest plan. Retain the original full desired/reference envelope.
    elapsed=(adapter.observe()['step']-adapter.mobile_program_start_step)//25
    if not np.array_equal(action,program.preview(elapsed)):
        raise ValueError('EXACT_PREDECLARED_FULL40_PROGRAM_RENEWAL')
    program.validate_envelope()
    return dict(original_full40_desired_and_reference_gate=True,renewal_cursor=elapsed,
                retiming=False,additional_smoothing=False)


@profiled('reference.full_H40_fixed_root_shadow_and_wire')
def prepare_install(adapter,action,original_doc):
    from scripts.pi0294.task_path_model import certify_chunk
    from .native_replay import forbid_integration
    import mujoco
    sample=adapter.measured_task_state();validate_preview(adapter,action,sample['tcp'])
    contact_rule=getattr(adapter,'native_preview_contact',None)
    contact_binding={} if contact_rule is None else {'contact_rule':contact_rule}
    if not getattr(adapter,'mobile_motion_armed',False):
        # Original stable startup, before any new mobile reference is allowed.
        from .target_reference import measured_base_hold
        doc=measured_base_hold(original_doc,adapter.base_pose_hold)
        shadow=certify_chunk(adapter.actual_state.model,sample,matrix10(action),stage='approach',
            history=adapter.reference_history[-2:]if len(adapter.reference_history)>=2 else(),
            **contact_binding)
        return doc,shadow
    from .mobile_safety_profile_v1 import require_armed
    require_armed(adapter)
    mode=adapter.mobile_mode
    doc=document(action,mode=mode,base_poses=getattr(adapter,'mobile_base_preview',None))
    try:
        with forbid_integration(mujoco):
            shadow=certify_chunk(adapter.actual_state.model,sample,matrix10(action),
                stage=getattr(adapter,'mobile_contact_stage','approach'),
                history=adapter.reference_history[-2:]if len(adapter.reference_history)>=2 else(),
                **contact_binding)
        shadow=dict(shadow,shadow_only=True)
    except ValueError as error:
        shadow=dict(status='FIXED_ROOT_IK_SHADOW_REJECT',reason=str(error),shadow_only=True,
                    physics_integrations=0,thresholds_changed=False)
    adapter.command_event(dict(phase='mobile-full40-and-shadow',native_step=sample['native_step'],
        mobile_mode=mode,state_dimension=7 if mode=='M_FREE'else 14,
        base_reference_present='base'in doc,fixed_root_IK=shadow))
    return doc,shadow
