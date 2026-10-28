"""Synthetic local contracts; these are not native Teacher/Eval evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'scripts/pi0294/replay_viewer'
spec = importlib.util.spec_from_file_location('replay_v1_test_package', PACKAGE/'__init__.py', submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from replay_v1_test_package.records import Episode, sha, add_offline_predictions, local_file
from replay_v1_test_package.server import handler
from replay_v1_test_package.kinematics import recorded_skeleton


def make(tmp_path, *, native=False):
    # 1x1 PNG, synthetic pixels. No actual camera data.
    import base64
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII=')
    rows = []
    camera_index = {}
    for i in range(3):
        pose = np.eye(4); pose[:3, 3] = [.1+i*.01, .2, .3]
        cam = {}
        for c in ('body_rgb', 'wrist_rgb'):
            name = f'images/{i:06d}-{c}.png' if native else f'{c}/{i:04d}.png'
            p = tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_bytes(png)
            cam[c] = dict(path=name, sha256=sha(p), timestamp_s=i*.05)
        if native:
            rows.append(dict(time_s=i*.05, tcp=pose.tolist(), object_pose=pose.tolist(), qpos=[i, 0], qvel=[0, 0], ctrl=[1], teacher_stage='fixture'))
            camera_index[str(i)] = cam
        else:
            rows.append(dict(policy_obs=dict(timestamp_s=i*.05, actual_tcp=pose.tolist(), actual_gripper=.2, **cam),
                             execution={}, privileged=dict(object_pose=pose.tolist(), teacher_stage='fixture'),
                             supervision=dict(desired=pose.tolist(), committed=pose.tolist(), actually_committed=True)))
    name = 'samples.jsonl' if native else 'samples.json'
    (tmp_path/name).write_text('\n'.join(json.dumps(r) for r in rows) if native else json.dumps(rows))
    (tmp_path/'audit.json').write_text(json.dumps({'status': 'SYNTHETIC_ONLY', 'task_success': False}))
    cfg = dict(id='synthetic', root=str(tmp_path), format='native_body' if native else 'fixed_base', source_kind='teacher',
               fixture=True, task='T1', samples_sha256=sha(tmp_path/name), audit_path='audit.json', audit_sha256=sha(tmp_path/'audit.json'))
    if native:cfg['native_camera_index'] = camera_index
    return cfg


def test_fixed_base_original_audit_and_source_unchanged(tmp_path):
    cfg=make(tmp_path); before={str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    e=Episode(cfg); payload=e.payload()
    assert len(payload['frames'])==3 and payload['fixture'] is True
    assert payload['original_audit']=={'status':'SYNTHETIC_ONLY','task_success':False}
    assert payload['frames'][0]['qpos'] is None
    assert payload['frames'][1]['actual_tcp'][0][3]==.11
    assert e.asset(0,'body_rgb')
    assert {str(p):sha(p) for p in tmp_path.rglob('*') if p.is_file()}==before


def test_native_and_no_invented_camera(tmp_path):
    cfg=make(tmp_path,native=True);e=Episode(cfg)
    assert e.frames[1]['qpos']==[1,0] and e.frames[1]['skeleton'] is None
    del cfg['native_camera_index'];e=Episode(cfg)
    assert e.frames[0]['cameras']['body_rgb'] is None


def test_clock_and_identity_refuse(tmp_path):
    cfg=make(tmp_path);rows=json.loads((tmp_path/'samples.json').read_text())
    rows[1]['policy_obs']['body_rgb']['timestamp_s']=.06
    (tmp_path/'samples.json').write_text(json.dumps(rows));cfg['samples_sha256']=sha(tmp_path/'samples.json')
    with pytest.raises(ValueError,match='CLOCK'):Episode(cfg)
    cfg['audit_sha256']='0'*64
    with pytest.raises(ValueError,match='AUDIT'):Episode(cfg)


def test_mutated_image_refuses_and_no_path_escape(tmp_path):
    e=Episode(make(tmp_path));(tmp_path/'body_rgb/0000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='RGB_IDENTITY'):e.asset(0,'body_rgb')
    with pytest.raises(ValueError):local_file(tmp_path,'../outside')
    with pytest.raises(ValueError):local_file(tmp_path,'/etc/passwd')


def test_highrate_preceding_only_and_no_future_state(tmp_path):
    cfg=make(tmp_path)
    # time+nq2+nv2+ctrl1+wrench6+safety1 = 13 columns
    rows=np.zeros((2,13));rows[:,0]=[.002,.052];rows[:,1]=[10,20]
    np.savez(tmp_path/'highrate.npz',rows=rows,nq=2,nv=2)
    cfg['highrate_sha256']=sha(tmp_path/'highrate.npz');e=Episode(cfg)
    assert e.frames[0]['qpos'] is None
    assert e.frames[1]['qpos']==[10.,0.] and e.frames[1]['state_time_s']==.002
    assert e.frames[2]['qpos']==[20.,0.]


def test_unexecuted_prediction_separate_no_relabelling(tmp_path):
    e=Episode(make(tmp_path));original=json.loads(json.dumps(e.frames))
    row=dict(frame_index=0,observation_time_s=0,executed=False,label='E1 fixture',xyz=[[.1,.2,.3]],model_identity='fixture',coordinate_frame='world')
    p=tmp_path/'predictions.json';p.write_text(json.dumps([row]));add_offline_predictions(e,dict(path=p.name,sha256=sha(p)))
    assert e.frames[0]['actual_tcp']==original[0]['actual_tcp'] and e.frames[0]['offline_predictions'][0]['executed'] is False
    row['executed']=True;p.write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='UNEXECUTED'):add_offline_predictions(e,dict(path=p.name,sha256=sha(p)))


def test_mj_forward_only_mock_actual_state_not_modified():
    class Model:
        nq=2;nv=2;nu=1;nbody=3;body_parentid=[0,0,1]
    class Data:
        def __init__(self,m):self.qpos=np.zeros(2);self.qvel=np.zeros(2);self.ctrl=np.zeros(1);self.xpos=np.zeros((3,3));self.time=0
    calls=[]
    class Module:
        MjData=Data
        @staticmethod
        def mj_forward(m,d):calls.append((d.qpos.copy(),d.time));d.xpos[:,0]=[0,d.qpos[0],d.qpos[1]]
        @staticmethod
        def mj_step(*args):raise AssertionError('NEVER_STEP')
    state=dict(qpos=[1,2],qvel=[0,0],ctrl=[.5],state_time_s=.4)
    before=json.dumps(state);out=recorded_skeleton(Model(),state,mujoco_module=Module)
    assert len(calls)==1 and out['new_integrations']==0 and out['edges']==[[0,1],[1,2]] and json.dumps(state)==before


def test_loopback_http_routes_and_rebinding_rejected(tmp_path):
    e=Episode(make(tmp_path))
    with ThreadingHTTPServer(('127.0.0.1',0),handler({e.id:e})) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            assert b'Episode Replay Viewer' in urllib.request.urlopen(url).read()
            assert json.load(urllib.request.urlopen(url+'/api/episodes'))[0]['fixture'] is True
            payload=json.load(urllib.request.urlopen(url+'/api/episode/synthetic'))
            assert payload['frames'][1]['time_s']==.05
            assert urllib.request.urlopen(url+'/api/asset/synthetic/0/body_rgb').read()==e.asset(0,'body_rgb')
            req=urllib.request.Request(url+'/api/episodes',headers={'Host':'evil.example'})
            with pytest.raises(urllib.error.HTTPError) as ex:urllib.request.urlopen(req)
            assert ex.value.code==403
        finally:server.shutdown();thread.join()


def test_explicit_sparse_source_indices_and_external_image_cache(tmp_path):
    import hashlib
    source=tmp_path/'source';source.mkdir();cfg=make(source)
    cache=tmp_path/'cache';cache.mkdir();mapping={}
    for p in source.rglob('*.png'):
        rel=p.relative_to(source).as_posix();name=p.parent.name+'_'+p.name;(cache/name).write_bytes(p.read_bytes());p.unlink()
        mapping[rel]={'root':'existing_cache','path':name}
    rows=json.loads((source/'samples.json').read_text())
    receipt={'source_frame_indices':[0,16,32],'source_frame_count':641,'original_samples_sha256':'a'*64,
             'selected_row_canonical_sha256':[hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() for r in rows]}
    (source/'selection.json').write_text(json.dumps(receipt))
    cfg.update(image_roots={'existing_cache':str(cache)},image_path_map=mapping,
        recorded_view={'kind':'recorded_keyframe_subset','source_frame_indices':[0,16,32],'source_frame_count':641,'original_samples_sha256':'a'*64,
                       'selection_receipt':{'path':'selection.json','sha256':sha(source/'selection.json')}})
    e=Episode(cfg)
    assert e.frames[1]['source_frame_index']==16 and e.frames[1]['cameras']['body_rgb']['explicit_local_mapping'] is True
    assert e.asset(1,'body_rgb') and e.payload()['recorded_view']['source_frame_count']==641
    cfg['recorded_view']['source_frame_indices']=[0,15,32]
    with pytest.raises(ValueError,match='SELECTION_IDENTITY'):Episode(cfg)
