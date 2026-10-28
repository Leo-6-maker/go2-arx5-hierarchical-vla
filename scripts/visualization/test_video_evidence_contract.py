"""Run directly with the existing Corelab NumPy/OpenCV CPU runtime."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
import cv2
import numpy as np
from eef_video_workflow import frame_index,matching_metrics,policy_class,scientific_key,sha,write,DEFAULT_ROOT,AUTH,admit_query
from bundle_adapter import guard_render_imports

def rejected(call):
    try:call()
    except (AssertionError,ValueError,RuntimeError):return
    raise AssertionError('Expected fail-closed rejection')

def main():
    checks=[]
    rgb=np.array([[[255,0,0],[0,0,255]]],dtype=np.uint8)
    assert cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).tolist()==[[[0,0,255],[255,0,0]]];checks.append('RGB_BGR')
    t=[0.,.02,.07]
    assert [frame_index(t,x) for x in [0,.019,.02,.069,.07,5]]==[0,0,1,1,2,2]
    assert frame_index([0,.02],.07)==1 # Unequal episode lengths hold last valid frame.
    rejected(lambda:frame_index([0,.02,.01],.03));checks.append('monotonic_source_time_frame_hold_unequal_duration')
    terminal=[{'time':.02,'post_reset':False},{'time':0.,'post_reset':True}]
    assert [x['time'] for x in terminal if not x['post_reset']]==[.02];checks.append('post_reset_pose_excluded_missing_terminal_not_invented')
    a=np.array([[1.,1.,0.],[0.,2.,0.]]);r=np.array([[1.,0.,0.],[0.,1.,0.]])
    dist,track=matching_metrics(a,r,np.zeros(3),[0,1],[0,1],'world','world')
    assert np.allclose(track,1) and not np.isclose(track.mean(),dist.min()-np.linalg.norm(r,axis=1).min())
    rejected(lambda:matching_metrics(a,r,np.zeros(3),[0,1],[0,2],'world','world'))
    rejected(lambda:matching_metrics(a,r,np.zeros(3),[0,1],[0,1],'native','world'));checks.append('same_tick_tracking_not_minimum_difference_frame_mismatch_rejected')
    # Reuse the public project's NumPy-only convention; no WBC/native constructor.
    from evidence_so3 import quat_to_matrix,matrix_to_quat,rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows
    q=np.array([0,0,np.sqrt(.5),np.sqrt(.5)]);rotation=quat_to_matrix(q)
    assert np.allclose(rotation@np.array([1,0,0]),[0,1,0])
    assert np.allclose(quat_to_matrix(matrix_to_quat(rotation)),rotation)
    assert np.allclose(rot6d_rows_to_matrix(rotation_matrix_to_rot6d_rows(rotation)),rotation)
    link=np.array([2.,3.,4.]);offset=np.array([.2,0,0]);tcp=link+rotation@offset
    assert np.allclose(tcp,[2,3.2,4]) and not np.allclose(tcp,link+offset)
    checks.append('nonzero_SE3_xyzw_native_row_rot6d_link6_TCP_separation')
    args=['tree','processor','statistics','decoder','source',['raw','processed','noise','S4',4,'runtime']]
    original=scientific_key(*args)
    for index in range(5):
        changed=args.copy();changed[index]+='changed';assert scientific_key(*changed)!=original
    changed=args.copy();changed[-1]=['raw','processed','different_seed_actual_noise','S4',4,'runtime'];assert scientific_key(*changed)!=original
    checks.append('cache_invalidated_by_tree_processor_statistics_decoder_source_actual_noise')
    admit_query(7,7,28);admit_query(102,95,408)
    for counts in ((8,0,32),(104,95,416),(96,96,384),(103,95,412)):
        rejected(lambda:admit_query(*counts))
    checks.append('valid_technical_attempt_and_head_caps_enforced_before_query')
    assert policy_class(1)=='ONE_CHUNK_POLICY_EXECUTION' and policy_class(3)=='RECEDING_POLICY_EXECUTION';checks.append('one_query_is_not_receding')
    block=guard_render_imports()
    try:
        for name in ('torch','transformers','isaacgym','isaacsim'):rejected(lambda:block.find_spec(name))
    finally:sys.meta_path.remove(block)
    checks.append('render_model_simulator_imports_blocked')
    entry=Path(__file__).with_name('eef_video_workflow.py')
    with tempfile.TemporaryDirectory() as folder:
        p=subprocess.run([sys.executable,str(entry),'--root',folder],capture_output=True,text=True)
        assert p.returncode==0 and json.loads(p.stdout)['new_physics_allowed']==0 and not list(Path(folder).iterdir())
        for authority in ('PI0256_PREPARATION',AUTH):
            p=subprocess.run([sys.executable,str(entry),'run-episode','--authorization',authority,'--root',folder],capture_output=True,text=True)
            assert p.returncode!=0 and 'GUARD_STUB_ONLY' in p.stderr and not list(Path(folder).iterdir())
    checks.append('default_readonly_and_PI0256_PI0257_episode_guard')
    write(DEFAULT_ROOT/'delivery/contract_test_receipt.json',dict(status='PASS',checks=checks,source_sha256=sha(Path(__file__)),observer_physical_proof='NOT_RUN; these CPU tests do not authorize a new recorder'))
    print(json.dumps(dict(status='PASS',checks=len(checks))),flush=True)

if __name__=='__main__':main()
