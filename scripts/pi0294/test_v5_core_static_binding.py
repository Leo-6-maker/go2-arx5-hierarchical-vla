"""CPU differential tests. None claims a native render, model or task PASS."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import successor_cd as s,successor_core_context as context
from scripts.pi0294.generalization import static_preview as preview
from scripts.pi0294.generalization import spec as S,validation as V


def bound(role=context.GEN_ROLE):
    marker=s.scope(1000.)
    limits=dict(CPU_s=1200,physics_ticks=0,policy_episodes=0,engine_starts=30,
                cpus=[5,6],operation_deadline_epoch=2000.)
    m=dict(scope=role,run_uuid='own',code=dict(vla_commit='a'*40),limits=limits)
    row=dict(status='RESERVED_NOT_SPENT',run_uuid='own',role=role,source_sha='a'*40,
             scope_grant_sha256=s.GRANT_SHA,**{k:limits[k]for k in ('CPU_s','physics_ticks','policy_episodes','engine_starts')})
    local=dict(V5_CD_Core_CPU_substages={context.GEN_ALLOCATION:row})
    return m,local,marker


def test_manifest_does_not_create_a_cpu_allocation():
    m,l,marker=bound();assert context.cpu_allocation(m,l,marker,now_epoch=1500.)
    with pytest.raises(PermissionError):context.cpu_allocation(m,{},marker,now_epoch=1500.)


@pytest.mark.parametrize('field,value',[('CPU_s',1201),('engine_starts',31),('physics_ticks',1),('policy_episodes',1),('cpus',[1,1]),('cpus',[0,1,2,3]),('CPU_s',True)])
def test_static_finite_substage_cannot_expand(field,value):
    m,l,marker=bound();m['limits'][field]=value
    with pytest.raises(PermissionError):context.cpu_allocation(m,l,marker,now_epoch=1500.)


@pytest.mark.parametrize('field,value',[('status','CLOSED'),('run_uuid','old'),('role','oldPV1'),('source_sha','b'*40),('scope_grant_sha256','0'*64)])
def test_actual_once_row_cannot_be_reused(field,value):
    m,l,marker=bound();l['V5_CD_Core_CPU_substages'][context.GEN_ALLOCATION][field]=value
    with pytest.raises(PermissionError):context.cpu_allocation(m,l,marker,now_epoch=1500.)


@pytest.mark.parametrize('now',[999.,2000.,2001.])
def test_real_task_window_and_stage_end(now):
    m,l,marker=bound()
    with pytest.raises(PermissionError):context.cpu_allocation(m,l,marker,now_epoch=now)


def packet():
    rows=[]
    for i in range(6):
        for visual in ('V0','V1'):
            layout=dict(generator_version=S.LAYOUT_VERSION,source_xy_m=[.44+.003*i,-.085],
                goal_xy_m=[.44+.003*i,.085],cube_yaw_rad=0.,direction='+Y',distance_bin='NEAR')
            rows.append(S.make_spec(layout,split='G_TRAIN',lineage_id='fixture-'+str(i),source_population_sha256='a'*64,
                physics_profile_sha256=S.digest(V.physics_profile()),seeds=dict(geometry=i,visual=7,reset=9),visual=visual))
    return dict(schema='GEN1_V5_PREVIEW_V2_BINDING_V1',version='GEN1_TRAIN_PREVIEW_V2_NEAR_015_018',specs=rows)


def test_exact_six_by_two_and_no_native_permission_from_metadata():
    p=packet();groups=preview.specifications(p);assert len(groups)==6
    assert all(row['qualification']==S.QUALIFICATION for pair in groups.values()for row in pair.values())
    assert p['specs'][0]['qualification']['dynamic_teacher']=='NOT_RUN'


@pytest.mark.parametrize('change',['missing','duplicate','wrong_reset','extra_layout'])
def test_preview_population_is_not_selected_by_outcome(change):
    p=packet()
    if change=='missing':p['specs'].pop()
    elif change=='duplicate':p['specs'][1]=deepcopy(p['specs'][0])
    else:
        row=p['specs'][1]
        if change=='wrong_reset':row['reset_profile']='GEN_COMMON_HOME'
        else:row['seeds']['reset']+=1
        row['scene_variant_id']=S.variant_id(row);row['design_hash']=S.digest({k:v for k,v in row.items()if k!='design_hash'})
    with pytest.raises((PermissionError,ValueError)):preview.specifications(p)


def test_xml_all_other_attributes_stay_bound():
    xml='<mujoco><worldbody><geom name="table" rgba=".69 .70 .66 1" friction=".8 .02 .002"/><geom name="cube_geom" rgba=".82 .04 .04 1"/></worldbody></mujoco>'
    a=preview.xml_invariants(xml)
    b=preview.xml_invariants(xml.replace('.69 .70 .66 1','.72 .65 .58 1'))
    assert a[0]==b[0] and a[1]!=b[1]
    assert preview.xml_invariants(xml.replace('.8 .02 .002','.9 .02 .002'))[0]!=a[0]
    assert preview.xml_invariants(xml.replace('.82 .04 .04 1','.0 .04 .04 1'))[0]!=a[0]


def reset():
    return dict(canonical_t0_state_sha256='a',all_other_XML_sha256='b',native_dimensions=dict(nq=15),
        arm_q=[1.],actual_initial_tcp=[[1.]],cameras={n:dict(K=[[1]],snapshot_id='variant/0')for n in preview.CAMERAS},
        raw_RGB_sha256={n:'c'for n in preview.CAMERAS},table_rgba_xml='0 0 0 1')


def test_color_only_ignores_card_id_but_keeps_real_camera_and_reset():
    a=reset();b=deepcopy(a);b['table_rgba_xml']='1 1 1 1'
    for n in preview.CAMERAS:b['cameras'][n]['snapshot_id']='other/0';b['raw_RGB_sha256'][n]='d'
    preview.compare_reset(a,b,same_visual=False)
    b['cameras']['wrist_rgb']['K']=[[2]]
    with pytest.raises(PermissionError):preview.compare_reset(a,b,same_visual=False)


def test_duplicate_render_and_physical_mutation_are_not_silently_accepted():
    a=reset();b=deepcopy(a);b['raw_RGB_sha256']['body_rgb']='changed'
    with pytest.raises(PermissionError):preview.compare_reset(a,b,same_visual=True)
    b=deepcopy(a);b['canonical_t0_state_sha256']='changed'
    with pytest.raises(PermissionError):preview.compare_reset(a,b,same_visual=False)


def test_native_imports_follow_full_entry_validation():
    import ast,inspect
    source=inspect.getsource(preview.run);tree=ast.parse(source);calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name):calls.append((node.lineno,node.func.id))
    validate=min(n for n,name in calls if name=='validate_runtime')
    imports=[n.lineno for n in ast.walk(tree)if isinstance(n,ast.Import)and any(a.name=='mujoco'for a in n.names)]
    assert imports and all(n>validate for n in imports)
    assert 'make_teacher'not in source and 'mj_step=prohibited_step'in source


def test_existing_core_entry_routes_static_role_only(monkeypatch,tmp_path):
    from scripts.pi0294.pv1 import core
    monkeypatch.setattr(core,'validate_runtime',lambda r,p:dict(scope=context.GEN_ROLE))
    monkeypatch.setattr(preview,'run',lambda r:dict(root=str(r),fixture=True))
    assert core.qualify(tmp_path,context.GEN_ALLOCATION)['fixture']
    with pytest.raises(PermissionError):core.qualify(tmp_path,'OLD_PILOT')
