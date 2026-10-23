"""CPU qualification of the existing private MuJoCo time-reference parser.

No ROS node, native controller, simulator, models or physical admission. This
uses the existing EEF10 converter and receiver, with the real frozen parser.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

PARSER_SHA='3fd59ca29b257a4fec2a66409d33099d0cc0befe4ef6a5039d80b44d03dadf90'

def _need(value,reason):
    if not value:raise PermissionError('TARGET_TIME_WIRE_CPU_'+reason)

def load_frozen_parser(path):
    """Hash before executing only the known pure parser, never its ROS node."""
    path=Path(path);raw=path.read_bytes()
    _need(hashlib.sha256(raw).hexdigest()==PARSER_SHA,'EXACT_FROZEN_PRIVATE_PARSER_BYTES')
    name='_dec1_frozen_mujoco_time_parser_cpu'
    _need(name not in sys.modules,'ONE_ISOLATED_PARSER_LOAD')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:exec(compile(raw,str(path),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module

def qualify_parser_file(path):
    """Synthetic wire tests against actual parser code; never task evidence."""
    import numpy as np
    from scripts.pi0294.base_intent import eef_document
    from scripts.pi0294.native_receiver import bind_packet
    from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows,quat_to_matrix
    parser=load_frozen_parser(path)
    a=np.zeros((40,10));a[:,:3]=[.4,-.2,.6];a[:,0]+=np.arange(40)*.001
    angle=.3;c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    a[:,3:9]=rotation_matrix_to_rot6d_rows(rotation);a[:,9]=np.linspace(.1,.9,40)
    original=a.copy();document=eef_document(a)
    document['_pi0294']=dict(episode='CPU_ONLY_NOT_NATIVE_EPISODE',sequence=0,
        source_mpc_s=10.,ttl_s=.4,operation='install')
    measured=parser.Pose.from_state([-.5,.5,.1,0.,0.,0.,1.])
    bound,source,sequence=bind_packet(document,episode='CPU_ONLY_NOT_NATIVE_EPISODE',sequence=-1,
        now=10.,trajectory_type=parser.TaskTrajectory,ee=measured,base=None)
    _need(source==10. and sequence==0 and bound.base is None,'WORLD_EEF_ONLY_NO_BASE_PADDING')
    _need(bound.ee.start_pose is None,'ACTUAL_ACTION0_NOT_MEASURED_POSE_SUBSTITUTION')
    times,poses,base=bound.window(10.,2.,.05)
    _need(base is None and len(times)==len(poses)==41,'FIRST_CLOSED_NATIVE_WORLD_WINDOW')
    _need(np.allclose(times,10.+np.arange(41)*.05,rtol=0,atol=1e-12),'ORIGINAL_20HZ_TWO_SECOND_GRID')
    for i,pose in enumerate(poses):
        _need(np.allclose(pose.position,a[min(i,39),:3],rtol=0,atol=1e-12),'NATIVE_WORLD_XYZ_AND_LAST_HOLD')
        _need(np.allclose(quat_to_matrix(pose.quaternion_xyzw),rotation,rtol=0,atol=1e-12),
            'NATIVE_ORIENTATION_ROWS_XYZW_NO_EXTRA_FRAME_ROTATION')
    _need(np.array_equal(a,original),'NO_ACTION_CORRECTION')
    rejection_cases=0
    for fault in ('stale','future','ttl','sequence','episode','wrong_frame','short_grid','legacy_top_level'):
        bad=deepcopy(document);now=10.
        if fault=='stale':now=10.025001
        elif fault=='future':now=9.999999
        elif fault=='ttl':bad['_pi0294']['ttl_s']=.5
        elif fault=='sequence':bad['_pi0294']['sequence']=1
        elif fault=='episode':bad['_pi0294']['episode']='OTHER'
        elif fault=='wrong_frame':bad['ee']['frame']='ee_start'
        elif fault=='short_grid':bad['ee']['waypoints'].pop()
        elif fault=='legacy_top_level':bad['waypoints']=bad['ee']['waypoints']
        rejected=False
        try:bind_packet(bad,episode='CPU_ONLY_NOT_NATIVE_EPISODE',sequence=-1,now=now,
            trajectory_type=parser.TaskTrajectory,ee=measured,base=None)
        except (ValueError,TypeError,KeyError):rejected=True
        _need(rejected,'ORIGINAL_RECEIVER_REJECT_'+fault);rejection_cases+=1
    cancel={'_pi0294':dict(document['_pi0294'],operation='cancel')}
    canceled,_,_=bind_packet(cancel,episode='CPU_ONLY_NOT_NATIVE_EPISODE',sequence=-1,now=10.,
        trajectory_type=parser.TaskTrajectory,ee=measured,base=None)
    _need(canceled is None,'EXPLICIT_CANCEL_PARSER_SEMANTICS')
    for column,value in ((0,float('nan')),(9,-.001),(9,1.001)):
        bad=a.copy();bad[39,column]=value;rejected=False
        try:eef_document(bad)
        except ValueError:rejected=True
        _need(rejected,'FULL_UNEXECUTED_H40_TAIL_REJECT');rejection_cases+=1
    return dict(status='CPU_ACTUAL_FROZEN_TARGET_TIME_WIRE_PASS',parser_sha256=PARSER_SHA,
        world_pose_samples_checked=41,invalid_cases_rejected=rejection_cases,
        measured_start_does_not_replace_action0=True,last_sample_held=True,
        no_BASE_field_created=True,explicit_cancel_parser_semantics=True,
        native_ACK_stop_or_physics_qualified=False,target_task_native_qualified=False,
        current_native_admission=False,models_loaded=0,physics_integrations=0,
        ROS_nodes_started=0,controller_changes=0,synthetic_wire_fixture_not_task_result=True)
