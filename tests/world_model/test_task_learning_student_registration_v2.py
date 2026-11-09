"""Registration seam fixtures; no claim of model, CUDA or native qualification."""
import copy,json,time
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_student_registration_v2 as registration
from go2arx5_vla.world_model import task_learning_student_v1 as student
from go2arx5_vla.world_model.task_learning_dev_v1 import DEV_OFFSETS
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id
from go2arx5_vla.world_model.storage import digest,write_json
from test_task_learning_student_v1 import panel as fixture_panel


def scenes(tmp_path):
    values={}
    for name in DEV_OFFSETS:
        root=tmp_path/name;root.mkdir();model=root/'SCENE.private.mjb';model.write_bytes(b'fixture-only-'+name.encode())
        path=root/'STATIC_RESULT.private.json'
        write_json(path,dict(layout=name,parent=scene_id(name),split='DEV',compiled=dict(scene_sha256=digest(model))))
        values[name]=registration.descriptor(path)
    return values


def test_candidate_endpoints_keep_all_original_dev_conditions_and_noise(tmp_path):
    static=scenes(tmp_path);p=fixture_panel()
    first=registration.make_panel('a'*40,2000,20261109,p['policies'],static,'candidate2000')
    next_=registration.make_panel('b'*40,4000,20261109,p['policies'],static,'candidate4000')
    assert len(student.validate(first))==len(student.validate(next_))==80
    assert all(a['query_seeds']==b['query_seeds']and a['model_sha256']==b['model_sha256']
        and a['method']==b['method']and a['layout']==b['layout']and a['restart']==b['restart']
        and a['native_run']!=b['native_run']for a,b in zip(first['assignments'],next_['assignments']))
    assert all(r['split']=='DEV'for r in first['assignments'])
    bad=copy.deepcopy(static);layout=next(iter(bad));path=Path(bad[layout]['path'])
    d=json.loads(path.read_bytes());d['split']='TEST';path.write_text(json.dumps(d));bad[layout]=registration.descriptor(path)
    with pytest.raises(ValueError,match='NOT_TEST'):
        registration.make_panel('a'*40,2000,20261109,p['policies'],bad,'candidate2000')


def test_base_closure_is_bound_to_the_actual_spec_and_both_exit_records(tmp_path):
    (tmp_path/'control').mkdir();spec=tmp_path/'EXECUTION_SPEC.json'
    write_json(spec,dict(registered_root=str(tmp_path)));pin=registration.descriptor(spec)
    lease=tmp_path/'control/LEASE.json';write_json(lease,dict(status='CLOSED',spec_sha256=pin['sha256'],worker_exit_code=0))
    write_json(tmp_path/'WORKER_EXIT.private.json',dict(exit_code=0))
    assert registration.closed_base(pin)['registered_root']==str(tmp_path)
    lease.write_text(json.dumps(dict(status='ACTIVE',spec_sha256=pin['sha256'],worker_exit_code=0)))
    with pytest.raises(ValueError,match='ACTUAL_CLOSED0'):registration.closed_base(pin)


def test_a_later_group_cannot_be_registered_without_actual_prefix_and_fresh_scopes(tmp_path):
    p=fixture_panel();p['registration']={k:{}for k in('source_files','native_base','gpu_base','scenes','memory_execution_reference')}
    p['registration'].update(source_root='/fixture',scene_allowlist='/fixture/scenes',scene_allowlist_sha256='a'*64,
        memory_execution='fixture-not-qualification')
    path=tmp_path/'PANEL.private.json';write_json(path,p)
    request=dict(panel=registration.descriptor(path),assignment_slice=[10,20],max_wall_s=3600)
    with pytest.raises((ValueError,TypeError,KeyError)):
        registration.group_input(request,p)
    request['assignment_slice']=[0,10]
    for label in('pilot1','fusion'):
        root=tmp_path/label;root.mkdir();binding=root/'BINDING.private.json'
        write_json(binding,dict(deadline_epoch=time.time()+7200));write_json(root/'LEASE.json',dict(status='ACTIVE'))
        request[label+'_binding']=registration.descriptor(binding)
    inp=registration.group_input(request,p)
    assert inp['assignment_slice']==[0,10]and 1800<inp['deadline_epoch']-time.time()<=3600
    (tmp_path/'fusion/LEASE.json').write_text(json.dumps(dict(status='CLOSED')))
    with pytest.raises(ValueError,match='FRESH_FINITE_DEV_SCOPE_fusion'):
        registration.group_input(request,p)


@pytest.mark.parametrize('launch_exit',[0,1])
def test_registered_group_uses_original_journal_and_never_replays_dispatch(tmp_path,monkeypatch,launch_exit):
    from types import SimpleNamespace
    from go2arx5_vla.world_model import pilot1_scope
    campaign=tmp_path/'campaign';(campaign/'control/journal').mkdir(parents=True)
    registry=tmp_path/'registry.json';write_json(registry,dict(entries=[]))
    bases={}
    for name in('native','gpu'):
        root=tmp_path/name;(root/'control').mkdir(parents=True)
        spec=root/'EXECUTION_SPEC.json'
        write_json(spec,dict(registered_root=str(root),campaign_root=str(campaign),registry=str(registry),
            scope='WM02_PILOT1_SIM_ONLY',owner='fixture',host='fixture',python='/fixture-never-executed'))
        bases[name]=registration.descriptor(spec)
        write_json(root/'control/LEASE.json',dict(status='CLOSED',spec_sha256=digest(spec),worker_exit_code=0))
        write_json(root/'WORKER_EXIT.private.json',dict(exit_code=0))
    panel=fixture_panel();folder=tmp_path/'panel';folder.mkdir();path=folder/'PANEL.private.json'
    panel['registration']=dict(native_base=bases['native'],gpu_base=bases['gpu'],source_root='/fixture-source',
        source_files={},scenes={},scene_allowlist='/fixture-allowlist',scene_allowlist_sha256='a'*64,
        memory_execution='fixture',memory_execution_reference={})
    write_json(path,panel)
    request=dict(panel=registration.descriptor(path),assignment_slice=[0,10],max_wall_s=3600,
        root=str(folder/'group-000-010'))
    for label in('pilot1','fusion'):
        folder_=tmp_path/label;folder_.mkdir();binding=folder_/'BINDING.private.json'
        write_json(binding,dict(deadline_epoch=time.time()+7200));write_json(folder_/'LEASE.json',dict(status='ACTIVE'))
        request[label+'_binding']=registration.descriptor(binding)
    monkeypatch.setattr(registration,'actual_site',lambda c:None)
    checked=[];monkeypatch.setattr(pilot1_scope,'validate',lambda s:checked.append(s))
    result=registration.register_group(request)
    assert result['status']=='DEV_GROUP_REGISTERED_NOT_DISPATCHED'and len(checked)==1
    assert len(json.loads(registry.read_bytes())['entries'])==1
    with pytest.raises(ValueError,match='NEVER_RECREATE'):registration.register_group(request)
    launches=[]
    def run(argv,**kwargs):
        if argv[0]=='systemctl':return SimpleNamespace(stdout='LoadState=not-found\nMainPID=0\n')
        launches.append(argv);return SimpleNamespace(returncode=launch_exit,stdout='',stderr='fixture')
    monkeypatch.setattr(registration.subprocess,'run',run)
    monkeypatch.setattr(registration.batch,'resources',lambda g:dict(fixture_only=True))
    input_path=Path(result['input']['path'])
    if launch_exit:
        with pytest.raises(ValueError,match='UNKNOWN_DEV_LAUNCH'):registration.dispatch_group(input_path)
    else:assert registration.dispatch_group(input_path)['exit_code']==0
    assert len(launches)==1 and 'CUDA_VISIBLE_DEVICES='in launches[0]
    assert '--property=Restart=no'in launches[0]and '--property=AllowedCPUs=0-3,8-11'in launches[0]
    assert (input_path.parent/'LAUNCH_INTENT.private.json').is_file()
    with pytest.raises(ValueError,match='UNKNOWN_DISPATCH'):registration.dispatch_group(input_path)
    assert len(launches)==1
