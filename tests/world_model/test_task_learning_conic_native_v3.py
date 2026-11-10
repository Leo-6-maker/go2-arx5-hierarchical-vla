import copy
import pytest
from go2arx5_vla.world_model import task_learning_conic_native_v3 as c
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id


def panel():
    rows=[]
    for i,(layout,method,seed)in enumerate(c.ORDER):
        rows.append(dict(native_run=f'new-{i}',layout=layout,method=method,parent=scene_id(layout),split='TRAIN',
            purpose=c.PURPOSE,restart=1,command=task.COMMAND,query_seeds=list(range(seed,seed+15)),model_sha256='a'*64))
    return dict(schema=c.SCHEMA,task_learning_role=c.PROTOCOL,protocol=task.protocol(c.PROTOCOL),B1_global_step=2000,
        preregistration={'sha256':c.CONTRACT_SHA},source_sha='b'*40,policies={m:{'arm':m}for m in c.METHODS},assignments=rows)


def test_fixed_population_full_denominators_and_old_namespace_rejects():
    p=panel();inp=dict(schema=c.INPUT_SCHEMA,assignment_slice=[0,4],population_predecessor=None,source_sha=p['source_sha'])
    assert batch.selection(inp,p)==p['assignments']
    report=c.summarize(p,[])
    assert report['requested']==4 and report['completed']==0 and not report['complete']
    assert all(v['requested']==2 and v['attempted']==0 for v in report['methods'].values())
    from go2arx5_vla.world_model.task_learning_local_policy_v1 import validate_panel as old_local
    from go2arx5_vla.world_model.task_learning_student_v1 import validate as dev
    for validate in(old_local,dev):
        with pytest.raises(ValueError):validate(p)
    for key,new in [('layout','TRAIN01'),('method','B1'),('query_seeds',list(range(15)))]:
        bad=copy.deepcopy(p);bad['assignments'][0][key]=new
        with pytest.raises(ValueError):c.validate_panel(bad)


def test_no_duplicate_outcome_or_partial_retry():
    p=panel();r=dict(p['assignments'][0],attempted=True,task_success=False,technical_valid=True,hard_safety_stop=False,
        actual_GR00T_queries=1,actual_native_integrations=5703)
    report=c.summarize(p,[r]);assert report['methods']['C500_RAW']['valid_task_failures']==1
    with pytest.raises(ValueError):c.summarize(p,[r,r])
    with pytest.raises(ValueError):c.selection(dict(schema=c.INPUT_SCHEMA,assignment_slice=[1,4],population_predecessor=None,source_sha=p['source_sha']),p)


def test_new_role_is_student_without_teacher_protocol_changes():
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert task.student_role(task.CONIC_ROLE)
    assert task.protocol(task.CONIC_ROLE)['expert_commands']=='NONE_STUDENT_NOT_EXPERT'
    assert task.protocol(task.CONIC_ROLE)['physics_dt_s']==.002
    assert teacher_fingerprint()==FROZEN_TEACHER


def test_admission_reads_actual_records_schema_and_rejects_missing_proof(monkeypatch):
    # The actual compare540 producer calls its per-output proof "records".
    req=dict(preregistration={'sha256':c.CONTRACT_SHA},Core_CPU540={'id':'cpu'},live_parity2={'id':'live'},package={'sha256':'x'})
    contract={'independent_native_panel':{'requested':4,'assignments':[dict(layout=l,method=m,query_seed_base=s)for l,m,s in c.ORDER]}}
    cpu=dict(status='PASS_ACTUAL_CORE_CPU540_TWICE_EXACT',matched=540,requested_unique=540,records=[{}]*540,
        all_bit_exact=True,actual_conic_fit_calls=1080,actual_solver_calls=2160)
    live=dict(requested_queries=2,actual_queries=2,rows=[{'index':0},{'index':3}],actual_model_loads=1,
        all_raw_bit_exact=True,all_conic_bit_exact=True,C500_package=req['package'],optimizer_updates=0,physics_integrations=0)
    monkeypatch.setattr(c,'pinned',lambda _:contract)
    monkeypatch.setattr(c,'closed_proof',lambda p,s:cpu if p==req['Core_CPU540']else live)
    monkeypatch.setattr(c,'verify_C500_package',lambda p,f:{'custody':'checked'})
    assert c.qualification(req,{})=={'custody':'checked'}
    cpu['records']=cpu['records'][:-1]
    with pytest.raises(ValueError):c.qualification(req,{})


def test_sealed_new_neural_dependencies_resolve_without_workspace_fallback(tmp_path):
    import subprocess,sys,shutil
    from pathlib import Path
    from go2arx5_vla.world_model.task_learning_live_pair_v1 import SOURCE_MEMBERS
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import SOURCE_MEMBERS as native
    root=Path(__file__).resolve().parents[2]
    capsule=tmp_path/'capsule';capsule.mkdir()
    for name in set(SOURCE_MEMBERS+native):
        old=root/name;target=capsule/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,target)
        for parent in old.parents:
            if parent==root:break
            init=parent/'__init__.py'
            if init.is_file():
                dst=capsule/init.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(init,dst)
    # The base capsule carries established contracts. New task modules must be
    # declared explicitly, so imports below cannot fall through to the worktree.
    package=root/'src/go2arx5_vla/contracts'
    shutil.copytree(package,capsule/'src/go2arx5_vla/contracts',dirs_exist_ok=True)
    shutil.copytree(root/'src/go2arx5_vla/trajectory',capsule/'src/go2arx5_vla/trajectory',dirs_exist_ok=True)
    for old in (root/'src/go2arx5_vla/world_model').glob('*.py'):
        if not old.name.startswith('task_learning_'):
            shutil.copyfile(old,capsule/'src/go2arx5_vla/world_model'/old.name)
    code='''import sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root)]
from go2arx5_vla.world_model.task_learning_local_analysis_v1 import geometry
from go2arx5_vla.world_model.task_learning_conic_policy_v3 import action_packet
import numpy as np
state=np.array([0.,0.,0.,1.,0.,0.,0.,1.,0.,1.])
raw=np.repeat(state[None],40,axis=0)
assert action_packet({'physical_H40':raw,'normalized':np.zeros((1,40,132))},'C500_RAW')['physical_H40'].shape==(40,10)
assert geometry(raw,state)['original_envelope_pass']
for name,module in list(sys.modules.items()):
 if name.startswith(('go2arx5_vla','scripts.'))and getattr(module,'__file__',None):
  assert Path(module.__file__).resolve().is_relative_to(root), (name,module.__file__)
assert not ({'torch','gr00t','scipy','clarabel'}&set(sys.modules))
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(capsule)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
