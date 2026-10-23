"""Task-blind CPU integration of the shared V2 mapper and target time wire.

The target-native prefix callback and event rows are CPU fixtures. No physical
qualification is claimed; actual private parser cross-check is separate.
"""
from array import array
from copy import deepcopy
import json
import numpy as np
import pytest
from . import target_time_transaction as unit
from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
from go2arx5_vla.adapters.execution_progress import ExecutionProgressAdapter,ProgressConfig,ActualFeedback,canonical_chunk_from_row_eef10
from go2arx5_vla.trajectory.so3 import quat_to_matrix,rotation_matrix_to_rot6d_rows

def fixture(mode='I_PROGRESS',controller_time=0.,rotating=False):
    raw=np.zeros((40,10));raw[:,:3]=[.4,0,.4];raw[:,0]+=np.arange(40)*.001
    roll=.1;orientation=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    raw[:,3:9]=rotation_matrix_to_rot6d_rows(orientation);raw[:,9]=.5
    if rotating:
        for i in range(40):
            angle=roll+.001*i;c,s=np.cos(angle),np.sin(angle)
            raw[i,3:9]=rotation_matrix_to_rot6d_rows(np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]]))
    angle=.2;tf=np.eye(4);tf[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    tf[:3,3]=[.02,.01,.03]
    gates=[];prefix=[]
    adapter=ExecutionProgressAdapter(ReferenceAdapterConfig('SOURCE_WORLD','TARGET_WORLD','TCP',stale_timeout_s=3.,frame_transform=tf),
        ProgressConfig(),mode=mode,episode_id='CPU_TARGET',validate_raw=lambda c:None,
        validate_mapped_full=lambda p:None,validate_prefix=lambda p:None)
    chunk=canonical_chunk_from_row_eef10(raw_eef10=raw,chunk_id='fixture',observation_timestamp=controller_time,point_timestamps=controller_time+np.arange(40)*.05,
        source_frame='SOURCE_WORLD',tcp_semantic_id='TCP',validate_raw_eef10=lambda a:None)
    transformed=chunk.transformed(tf,target_frame='TARGET_WORLD')
    feedback=ActualFeedback(controller_time,transformed.eef_position[0],transformed.eef_orientation[0],.5,'TARGET_WORLD','TCP',None)
    adapter.load_chunk(chunk,feedback,controller_timestamp=controller_time)
    proposal=adapter.propose(feedback,controller_timestamp=controller_time)
    mapped_state=np.r_[transformed.eef_position[0],rotation_matrix_to_rot6d_rows(quat_to_matrix(transformed.eef_orientation[0])),.5]
    args=dict(expected_raw_sha256=unit.raw_hash(raw),raw_actual_state10=raw[0],mapped_actual_state10=mapped_state,
        episode='CPU_TARGET',sequence=0,source_mpc_s=controller_time,expected_world_frame='TARGET_WORLD',expected_tcp='TCP',
        validate_target_native_prefix=lambda p:prefix.append(p.command_id),record_gate=gates.append)
    return raw,proposal,args,gates,prefix,transformed

def events(prepared):
    doc=json.loads(prepared.wire_json);points=doc['ee']['waypoints'];source=prepared.source_mpc_s
    return [dict(kind='accept',ref_id=prepared.wire_sha256,raw=prepared.wire_json,source_mpc_s=source,receiver_mpc_s=source,deadline=source+.4),
        dict(kind='installed',ref_id=prepared.wire_sha256,bound_start=source,observation_time_s=source,
            times=[source+p['time'] for p in points],poses=[array('f',p['position']+p['quaternion_xyzw']).tolist() for p in points],inputs=[[0.]*12 for _ in points])]

@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_shared_V2_transform_once_then_existing_target_EEF_only_wire_and_actual_pose_binding(mode):
    raw,p,args,gates,prefix,target=fixture(mode);before=raw.copy();prepared=unit.prepare(raw,p,**args)
    wire=json.loads(prepared.wire_json)
    assert set(wire)=={'version','ee','_pi0294'} and len(wire['ee']['waypoints'])==41
    # Frozen shared preview has a .01m first lookahead. Preserve that already
    # mapped intent, rather than substituting the raw/achieved first pose.
    direction=(target.eef_position[1]-target.eef_position[0])/.001
    np.testing.assert_allclose(wire['ee']['waypoints'][0]['position'],target.eef_position[0]+.01*direction,rtol=0,atol=1e-12)
    np.testing.assert_allclose(wire['ee']['waypoints'][0]['position'],p.mapped_full.eef_position[0],rtol=0,atol=1e-12)
    assert [g['kind'] for g in gates]==['RAW_H40','MAPPED_H40'] and prefix==[p.command_id]
    np.testing.assert_array_equal(raw,before)
    ack=unit.require_install_ack(prepared,events(prepared))
    assert not ack['physics_advance_authorized'] and not ack['target_task_native_qualified']

@pytest.mark.parametrize('fault',['identity','raw_tail','mapped_tail','working_gripper','missing_native','false_native','source_receipt','clock','frame','tcp'])
def test_no_unchecked_full_tail_or_source_shadow_can_become_target_packet(fault):
    raw,p,args,gates,prefix,_=fixture()
    if fault=='identity':args['expected_raw_sha256']='0'*64
    elif fault=='raw_tail':raw[39,0]=5.;args['expected_raw_sha256']=unit.raw_hash(raw)
    elif fault=='mapped_tail':
        p.mapped_full.eef_position.setflags(write=True);p.mapped_full.eef_position[39,0]=5.
    elif fault=='working_gripper':
        p.mapped_full.gripper.setflags(write=True);p.mapped_full.gripper[39]=0.
    elif fault=='missing_native':args['validate_target_native_prefix']=None
    elif fault=='false_native':args['validate_target_native_prefix']=lambda p:False
    elif fault=='source_receipt':args['validate_target_native_prefix']=lambda p:{'source_S1':'PASS','shadow_integrations':200}
    elif fault=='clock':args['source_mpc_s']=-.025001
    elif fault=='frame':args['expected_world_frame']='OTHER_WORLD'
    elif fault=='tcp':args['expected_tcp']='OTHER_TCP'
    with pytest.raises((PermissionError,ValueError)):unit.prepare(raw,p,**args)
    assert prefix==[]

@pytest.mark.parametrize('fault',['no_accept','no_install','reject','other_hash','clock','ttl','bound_start','pose','BASE','input','short_window','grid','one_native_ulp'])
def test_missing_or_wrong_native_install_evidence_cannot_release_physics(fault):
    raw,p,args,*_=fixture();prepared=unit.prepare(raw,p,**args);rows=events(prepared)
    if fault=='no_accept':rows.pop(0)
    elif fault=='no_install':rows.pop()
    elif fault=='reject':rows.append(dict(kind='reject',wire_sha256=prepared.wire_sha256))
    elif fault=='other_hash':rows[1]['ref_id']='0'*64
    elif fault=='clock':rows[0]['receiver_mpc_s']=.026
    elif fault=='ttl':rows[0]['deadline']=.5
    elif fault=='bound_start':rows[1]['bound_start']=.01
    elif fault=='pose':rows[1]['poses'][5][0]+=.001
    elif fault=='BASE':rows[1]['poses']=[r+[0.]*7 for r in rows[1]['poses']]
    elif fault=='input':rows[1]['inputs'][0][0]=1.
    elif fault=='short_window':rows[1]['times'].pop()
    elif fault=='grid':rows[1]['times'][5]+=.001
    elif fault=='one_native_ulp':rows[1]['poses'][5][0]=float(np.nextafter(np.float32(rows[1]['poses'][5][0]),np.float32(np.inf)))
    with pytest.raises(PermissionError):unit.require_install_ack(prepared,rows)

@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
@pytest.mark.parametrize('receiver_delay',[0.,.0125,.025])
@pytest.mark.parametrize('rotating',[False,True])
def test_exact_private_parser_window_matches_prepared_world_reference(mode,receiver_delay,rotating,monkeypatch):
    """Actual frozen parser CPU math, not invented native ACK evidence."""
    import os,sys
    from .target_time_wire_cpu import load_frozen_parser
    from scripts.pi0294.native_receiver import bind_packet
    path=os.environ.get('PI0294_FROZEN_TARGET_PARSER')
    if not path:pytest.skip('private frozen parser requires registered runtime bytes')
    # Scope the loader's module cache to this test; do not leak it into other
    # private-parser qualifications in the same pytest process.
    name='_dec1_frozen_mujoco_time_parser_cpu'
    monkeypatch.setitem(sys.modules,name,None)
    monkeypatch.delitem(sys.modules,name)
    parser=load_frozen_parser(path)
    raw,p,args,*_=fixture(mode,rotating=rotating)
    prepared=unit.prepare(raw,p,**args);wire=json.loads(prepared.wire_json)
    bound,source,sequence=bind_packet(wire,episode=prepared.episode,sequence=-1,
        now=receiver_delay,trajectory_type=parser.TaskTrajectory,
        ee=parser.Pose.from_state([0.,0.,0.,0.,0.,0.,1.]),base=None)
    times,poses,base=bound.window(receiver_delay,2.,.05)
    assert base is None and source==prepared.source_mpc_s and sequence==prepared.sequence
    rows=events(prepared)
    rows[0]['receiver_mpc_s']=receiver_delay
    rows[1].update(observation_time_s=receiver_delay,times=times,
        poses=[array('f',pose.as_state()).tolist() for pose in poses],inputs=[[0.]*12 for _ in times])
    result=unit.require_install_ack(prepared,rows)
    assert result['status']=='BOUND_ORIGINAL_MUJOCO_RECEIVER_INSTALL_ACK'
    assert not result['physics_advance_authorized']


def test_native_float32_serialization_preserves_original_raw_reference_without_widened_tolerance():
    raw,p,args,*_=fixture();prepared=unit.prepare(raw,p,**args);rows=events(prepared)
    original=json.loads(prepared.wire_json)['ee']['waypoints'][0]['position']
    assert rows[1]['poses'][0][:3]!=original  # Real native float32 round trip.
    assert unit.require_install_ack(prepared,rows)['wire_sha256']==prepared.wire_sha256
    rows[1]['poses'][0][:3]=original
    with pytest.raises(PermissionError):unit.require_install_ack(prepared,rows)
