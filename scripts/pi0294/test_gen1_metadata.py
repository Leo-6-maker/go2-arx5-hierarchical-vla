"""Synthetic CPU rules plus saved public geometry; no native qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.pi0294.generalization import spec as S, population as P, validation as V
from scripts.pi0294.scene_cards import INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/pi0294/generalization/gen1.yaml'


def example(split='G_TRAIN', x=.45, lineage='fixture-parent', visual='V0', reset='LEGACY_ABOVE_SOURCE'):
    layout = dict(generator_version=S.LAYOUT_VERSION, source_xy_m=[x,-.085], goal_xy_m=[x,.085],
                  cube_yaw_rad=0., direction='+Y', distance_bin='NEAR')
    return S.make_spec(layout, split=split, lineage_id=lineage, source_population_sha256='a'*64,
                       physics_profile_sha256=S.digest(V.physics_profile()),
                       seeds=dict(geometry=1, visual=2, reset=3), visual=visual, reset=reset)


def resign(value):
    value['scene_variant_id']=S.variant_id(value)
    value['design_hash'] = S.digest({k:v for k,v in value.items() if k!='design_hash'})
    return value


def test_seeds_order_and_visual_independence():
    c = V.load_config(CONFIG)
    first = P.candidate_pool(c['new_slots'], geometry_master_seed=c['seeds']['geometry'])
    second = P.candidate_pool(list(reversed(c['new_slots'])), geometry_master_seed=c['seeds']['geometry'])
    assert first == second and len(first['candidates']) == 768
    assert len({S.seed(1, s, 'same') for s in ('geometry','visual','reset')}) == 3
    a,b = example(),example(visual='V1')
    assert a['geometry_id'] == b['geometry_id'] and a['design_hash'] != b['design_hash']
    b['seeds']['visual'] = 100
    S.validate(resign(b));assert a['geometry_id'] == b['geometry_id']


@pytest.mark.parametrize('key,value', [('cube_yaw_rad',float('nan')),('cube_yaw_rad',True),
                                      ('source_xy_m',[450.,-85.]),('goal_xy_m',[.45,float('inf')]),
                                      ('direction','X'),('distance_bin','FAR'),('mass_kg',1.)])
def test_bad_units_nonfinite_unknown_fields(key,value):
    s=example();s['layout'][key]=value
    with pytest.raises(ValueError):S.validate(s)


def test_no_metadata_authority_and_no_physics_injection():
    s=example();s['qualification']['dynamic_teacher']='PASS'
    with pytest.raises(ValueError):S.validate(resign(s))
    s=example();s['physics']={'mass':1.}
    with pytest.raises(ValueError):S.validate(resign(s))
    for split in S.SPLITS:
        with pytest.raises((ValueError,PermissionError)):V.require_training_export(example(split))


def test_original_task_pair_and_table_only_variants():
    a,b=S.design_cards(example())
    assert a['instruction']==INSTRUCTIONS['T1'] and b['instruction']==INSTRUCTIONS['T2']
    assert a['initial_state_sha256']==b['initial_state_sha256']
    for key in ('source_xy','goal_xy','cube_yaw','visual'):assert a[key]==b[key]
    warm=S.design_cards(example(visual='V1'))[0]
    assert {k for k in a['visual'] if a['visual'][k]!=warm['visual'][k]}=={'table_rgba'}
    assert not a['native_execution_authorized'] and a['split']=='G_TRAIN'


def test_common_home_is_independent_unqualified_version():
    p=V.common_home_preparation()
    assert p['selected'] is None and p['status']=='COMMON_HOME_UNQUALIFIED'
    assert len(p['candidates'])==3 and not p['selection_uses_TEST']
    for s in [example(reset='GEN_COMMON_HOME'),example(x=.46,reset='GEN_COMMON_HOME')]:
        with pytest.raises(ValueError,match='COMMON_HOME_UNQUALIFIED'):S.design_cards(s)
    assert p==V.common_home_preparation()


def test_saved_COV1_ranges_and_original_lineage_preserved():
    path=ROOT/'reports/PI0294/DATA_COV1/DATA_COV1_FROZEN_POPULATION.json'
    groups=[g for g in json.loads(path.read_bytes())['P24_groups'] if g['replicate']==1]
    from scripts.pi0294.data_cov1.plan import preflight_score_geometry
    rejected=0
    for g in groups:
        s=S.import_cov1_train(g,source_population_sha256='a'*64,physics_profile_sha256='b'*64,
                             visual_master_seed=10,reset_master_seed=11)
        assert s['lineage_id']==g['group_id'] and s['layout']['distance_bin']==g['distance_bin']
        assert s['layout']['source_xy_m']==g['source_xy']
        try:preflight_score_geometry(g)
        except PermissionError:rejected+=1
        s['study_split']='G_TEST'
        with pytest.raises(ValueError):S.validate(resign(s))
    assert len(groups)==12 and rejected==6


def test_transitive_lineage_not_just_greedy_or_visual():
    rows=[example('G_TRAIN',.450,'a'),example('G_TRAIN',.4515,'b','V1'),example('G_TEST',.453,'c')]
    r=P.audit_splits(rows)
    assert len(r['cross_split_components'])==1 and len(r['lineage_components'][0]['members'])==3
    with pytest.raises(ValueError,match='CROSS_SPLIT'):P.require_separated_metadata(r)
    r=P.audit_splits([example('G_TRAIN',.44,'same'),example('G_TEST',.46,'same')])
    assert r['cross_split_components']


def test_historical_bridge_is_in_transitive_component():
    history=[dict(group_id='old',split='TRAIN',source_xy=[.4515,-.085],goal_xy=[.4515,.085],cube_yaw=0.)]
    r=P.audit_splits([example('G_TRAIN',.45,'a'),example('G_TEST',.453,'b')],historical_groups=history)
    assert r['cross_split_components']
    history[0]['cube_yaw']=True
    with pytest.raises(ValueError):P.audit_splits([example()],historical_groups=history)


def test_missing_history_and_even_all_hashes_do_not_grant_qualification():
    for sources in ({},{k:['a'*64] for k in P.HISTORY_FAMILIES}):
        r=P.audit_splits([example()],historical_source_sha256=sources)
        assert not r['new_TEST_qualified'] and not r['history_completeness_qualified']
        assert not r['training_export_authorized']


def test_proposed_combination_leak_is_rejected():
    s=example();s['layout']['direction']='-Y';s['layout']['source_xy_m'][1]=.085;s['layout']['goal_xy_m'][1]=-.085
    s['geometry_id']=S.geometry_id(s['layout']);s['visual_profile']='V1';resign(s)
    r=P.audit_splits([s]);assert r['proposed_holdout_leaks_into_TRAIN_or_DEV']
    with pytest.raises(ValueError,match='HOLDOUT'):P.require_separated_metadata(r)


def test_finite_selection_exhaustion_keeps_failure_and_no_redraw():
    pool=P.candidate_pool([dict(slot_id='GEN1/G_TEST/+Y/NEAR/00',stratum='+Y/NEAR')],geometry_master_seed=1,candidates_per_slot=1)
    row=pool['candidates'][0];l=row['layout']
    old=[dict(group_id='old',split='TRAIN',source_xy=l['source_xy_m'],goal_xy=l['goal_xy_m'],cube_yaw=l['cube_yaw_rad'])]
    result=P.select_metadata(pool,old)
    assert result['exhausted_slots'] and not result['selected'] and len(result['rejected'])==1
    assert pool['pool_sha256']==S.digest(pool['candidates'])


def test_clean_process_no_model_engine_or_network_and_hash_seed_independence():
    code='''
import sys,importlib.abc,json
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('mujoco','torch','rclpy','OpenGL','gr00t') or fullname.endswith('.engine'):
   raise AssertionError('FORBIDDEN_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
def guard(event,args):
 if event in ('socket.connect','subprocess.Popen','os.system'):raise AssertionError(event)
sys.addaudithook(guard)
from scripts.pi0294.generalization.validation import load_config,common_home_preparation
from scripts.pi0294.generalization.population import candidate_pool
c=load_config('configs/pi0294/generalization/gen1.yaml')
p=candidate_pool(c['new_slots'],geometry_master_seed=c['seeds']['geometry'])
assert common_home_preparation()['selected'] is None
print(p['pool_sha256'])
'''
    outputs=[]
    for h in ('1','932'):
        env=dict(os.environ,PYTHONHASHSEED=h,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'))
        p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        outputs.append(p.stdout)
    assert outputs[0]==outputs[1]


def test_artifact_builder_and_predeclared_preview_have_no_native_effects():
    from scripts.pi0294.generalization import artifacts as A
    path=ROOT/'reports/PI0294/DATA_COV1/DATA_COV1_FROZEN_POPULATION.json'
    groups=[g for g in json.loads(path.read_bytes())['P24_groups'] if g['replicate']==1]
    c=V.load_config(CONFIG)
    args=dict(train_groups=groups,train_source_sha256='a'*64,history=groups,history_sources={})
    a=A.build_metadata(c,**args)
    args['train_groups']=list(reversed(groups));b=A.build_metadata(c,**args)
    assert a==b and len(a['manifest'])==36
    assert not a['split_report']['new_TEST_qualified']
    assert a['split_report']['status']=='EXCLUSION_INDEX_INCOMPLETE'
    p=A.preview_design(a['manifest'])
    assert len(p['geometry_ids'])==6 and p['expected_initial_images']==24 and p['actual_images']==0
    assert not p['native_execution_authorized'] and not p['Teacher_pilot_authorized']
    assert not a['selection']['exhausted_slots']


@pytest.mark.parametrize('edit', ['physics','camera','table','slots'])
def test_config_cannot_silently_expand_frozen_dimensions(tmp_path,edit):
    c=json.loads(CONFIG.read_bytes())
    if edit=='physics':c['physics_profile_sha256']='f'*64
    if edit=='camera':c['camera_profile']='MOVING_CAMERA'
    if edit=='table':c['visual_profiles']['V0'][3]=.5
    if edit=='slots':c['new_slots'].pop()
    p=tmp_path/'gen1.yaml';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):V.load_config(p)
