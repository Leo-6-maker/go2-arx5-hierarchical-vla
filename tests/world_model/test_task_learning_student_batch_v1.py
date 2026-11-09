"""Finite pair orchestration fixtures; no actual CUDA, worker or physics claim."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
from go2arx5_vla.world_model.storage import digest,write_json
from scripts.pi0294.e1.journal import Journal
from test_task_learning_student_v1 import panel


def test_group_keeps_the_fixed_interleaved_population_and_matched_pairs():
    p=panel();inp=dict(schema=batch.SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,40])
    assert batch.selection(inp,p)==p['assignments'][:40]
    for span in([1,40],[0,41],[-2,40],[40,40],[0,82],[False,40]):
        with pytest.raises(ValueError):batch.selection(dict(inp,assignment_slice=span),p)
    with pytest.raises(ValueError):batch.selection(dict(inp,source_sha='wrong'),p)


def setup(tmp_path):
    registered={};specs={}
    for name in('native','gpu'):
        root=tmp_path/name;root.mkdir();(root/'logs').mkdir();(root/'tmp').mkdir()
        s=dict(registered_root=str(root),source_root=str(tmp_path),native_environment={'CUDA_VISIBLE_DEVICES':''},
            GPU_uuid='fixture-only-GPU',python='/fixture-python-never-executed',deadline_epoch=10**11)
        p=root/'EXECUTION_SPEC.json';write_json(p,s);registered[name]=dict(path=str(p),sha256=digest(p));specs[name]=s
    return registered,specs,Journal(tmp_path/'journal','fixture')


def test_both_workers_start_once_with_separate_device_masks_and_existing_entries_never_replay(tmp_path,monkeypatch):
    registered,specs,j=setup(tmp_path);calls=[]
    class Child:
        pid=98765;returncode=0
        def __init__(self,gpu):self.gpu=gpu;self.reads=0
        def poll(self):
            self.reads+=1
            return None if self.gpu and self.reads==1 else 0
    def start(cmd,**kw):
        calls.append((cmd,kw));return Child('pilot1_neural_worker'in cmd[3])
    monkeypatch.setattr(batch.subprocess,'Popen',start)
    assert batch.execute_pair(registered,j,tmp_path,0)==dict(gpu=0,native=0)
    assert len(calls)==2
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']=='fixture-only-GPU'
    assert calls[1][1]['env']['CUDA_VISIBLE_DEVICES']==''
    with pytest.raises(ValueError,match='RECONCILE_EXISTING_OR_UNKNOWN'):
        batch.execute_pair(registered,j,tmp_path,0)
    assert len(calls)==2


def test_immediate_serving_failure_starts_no_native_worker_and_retains_dispatch(tmp_path,monkeypatch):
    registered,specs,j=setup(tmp_path);calls=[]
    class Failed:
        pid=98765;returncode=1
        def poll(self):return 1
    def start(cmd,**kw):calls.append(cmd);return Failed()
    monkeypatch.setattr(batch.subprocess,'Popen',start)
    with pytest.raises(ValueError,match='GPU_EXIT_BEFORE_NATIVE_START'):
        batch.execute_pair(registered,j,tmp_path,0)
    assert len(calls)==1 and not j.path('NATIVE-RUN').exists()
    with pytest.raises(ValueError,match='RECONCILE_EXISTING_OR_UNKNOWN'):
        batch.execute_pair(registered,j,tmp_path,0)
    assert len(calls)==1


def test_new_pair_removes_teacher_and_old_pause_bindings_and_keeps_original_b0_processor(tmp_path):
    p=panel();row=p['assignments'][0]
    path=tmp_path/'STATIC.json';write_json(path,dict(layout=row['layout'],parent=row['parent'],
        compiled=dict(scene_sha256=row['model_sha256'])))
    inp=dict(source_root=str(tmp_path/'source'),source_sha=p['source_sha'],source_files={},
        scenes={row['layout']:dict(path=str(path),sha256=digest(path))},pilot1_binding='/fixture/new-P1',
        pilot1_binding_sha256='a'*64,fusion_binding='/fixture/new-Fusion',fusion_binding_sha256='b'*64,
        panel='/fixture/fixed-DEV80',panel_sha256='c'*64,scene_allowlist='/fixture/scenes',
        scene_allowlist_sha256='d'*64,memory_execution='pending-actual-qualification',
        memory_execution_reference=dict(path='/fixture/not-an-actual-proof',sha256='e'*64))
    base=dict(native_environment={},scope='WM02_PILOT1_SIM_ONLY',task_learning_teacher_mode='NORMAL_TEACHER_V7',
        target_gr00t_collection={'stale':True},paused_query_pair='/fixture/stale-pair',night1_binding='/fixture/expired')
    gbase=dict(scope='WM02_FUSION_REFERENCE_SIM_ONLY',neural_bindings={'source':'/fixture/vendor'},
        checkpoint='/fixture/original-parent',recorded_observation='/fixture/stale-capture',night1_binding='/fixture/expired')
    n,g=batch.specs(inp,p,row,base,gbase,tmp_path/'native',tmp_path/'gpu',10**11)
    assert not any(k in n for k in('task_learning_teacher_mode','target_gr00t_collection','paused_query_pair','night1_binding'))
    assert not any(k in g for k in('recorded_observation','night1_binding'))
    assert g['checkpoint']=='/fixture/original-parent'and g['neural_bindings']==gbase['neural_bindings']
    assert n['target_gr00t_policy']==g['target_gr00t_policy']==p['policies']['B0']
    assert n['target_gr00t_query_seeds']==g['target_gr00t_query_seeds']==row['query_seeds']
    assert n['data_role']=='DEV'and g['physics']is False and n['cpus']==[0,1,2,3]and g['cpus']==[8,9,10,11]
