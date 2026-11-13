import copy
import hashlib
import json

import numpy as np
import pytest

from go2arx5_vla.sla_adapter import named_dataset_v5 as named
from go2arx5_vla.sla_adapter.dense_dataset_v4 import manifest_contract as old_contract
from go2arx5_vla.sla_adapter.dense_recipe_v4 import visit_map
from go2arx5_vla.world_model.corelab_expert_semantics_v3 import ROLE
from scripts.sla.admit_named_target_v5 import source_episode


def cohort():
    return dict(schema=named.SCHEMA,state='committed',dataset_id=named.DATASET_ID,scope=named.SCOPE,
        tasks=['PICK_HOLD'],objects=1,split='TRAIN',codec=named.VERSION,D100_new=False,D300_new=False,
        PICK_PLACE_admitted=False,legacy_engineering_receipts_modified=False,old_2EP_reused=0,
        old_engineering16_reused=0,phase_input_to_student=False,consumption_proven=True,
        actual_success_tail_s=2.,command_period_ticks=25,frame_period_ticks=50,
        anchor_period_ticks=250,physics_dt_s=.002,supervised_windows=1220,
        selection_rule='FIRST20_DECLARED_ASSIGNMENTS_ALL_COMPLETE_NO_SUBSTITUTION',
        episodes=[dict(id=f'{layout}-r{restart}',layout=layout,restart=restart,split='TRAIN',
            task='PICK_HOLD',prefix_atoms=10,windows=61,frames=321,commands=640,native_integrations=21703)
            for layout,restart in named.CONDITIONS])


def test_separate_typed_reader_does_not_relax_two_episode_contract():
    value=cohort()
    named.NamedTargetDataset.validate_manifest(value)
    with pytest.raises(ValueError):old_contract(value)
    value['episodes']=value['episodes'][:2]
    with pytest.raises(ValueError):named.manifest_contract(value)


@pytest.mark.parametrize('mutation',['duplicate','substitute','old_restart','D100','placement','future',
                                    'student_phase','incomplete_tail','missing_frame','wrong_clock'])
def test_no_old_data_or_success_based_substitution(mutation):
    value=cohort()
    if mutation=='duplicate':value['episodes'][1]=copy.deepcopy(value['episodes'][0])
    elif mutation=='substitute':value['episodes'][0],value['episodes'][1]=value['episodes'][1],value['episodes'][0]
    elif mutation=='old_restart':value['episodes'][0]['restart']=1
    elif mutation=='D100':value['D100_new']=True
    elif mutation=='placement':value['PICK_PLACE_admitted']=True
    elif mutation=='future':value['consumption_proven']=False
    elif mutation=='student_phase':value['phase_input_to_student']=True
    elif mutation=='incomplete_tail':value['actual_success_tail_s']=0
    elif mutation=='missing_frame':value['episodes'][0]['frames']=320
    else:value['command_period_ticks']=50
    with pytest.raises(ValueError):named.manifest_contract(value)


def test_original_sampler_uses_new_episodes_and_every_phase_without_phase_input():
    class Dataset:
        manifest_sha='b'*64
        def catalog_rows(self):
            return [dict(split='TRAIN',task='PICK_HOLD',episode=ep['id'],phase=phase,anchor=i)
                for ep in cohort()['episodes'] for i,phase in enumerate(
                    ['PREGRASP']*25+['DESCEND']*15+['CLOSE']*10+['LIFT']*5+['HOLD']*6)]
    rows=Dataset().catalog_rows()
    visits,binding=visit_map(Dataset())
    assert len(rows)==1220 and visits.shape==(96000,)
    assert set(visits)==set(range(1220))
    assert len({rows[i]['episode'] for i in visits})==20
    assert not binding['student_phase_input'] and binding['planned_source_visits']==0
    assert all(18000<sum(rows[i]['phase']==p for i in visits)<20500
               for p in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source(tmp_path):
    root=tmp_path/'fresh-train01-r5'
    row=dict(native_run=root.name,parent='TASK_LEARNING_UNDER_HAND_TRAIN01',restart=5,
             prefix_atoms=10,model_sha256='c'*64)
    panel=dict(path=str(tmp_path/'PANEL.private.json'),sha256='d'*64)
    s=dict(registered_root=str(root),task_learning_role=ROLE,scene_id=row['parent'],
        expert_cadence_restart=5,task_learning_grasp=dict(prefix_atoms=10),
        stable_task_panel=panel['path'],stable_task_panel_sha256=panel['sha256'],source_sha='e'*40)
    write(root/'EXECUTION_SPEC.json',s)
    write(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,
                                      spec_sha256=pin(root/'EXECUTION_SPEC.json')['sha256']))
    write(root/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    result=dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,physics_stops=[],
                actual_target_integrations=21703,reference_qualification=dict(task_condition_success=True))
    write(root/'native-01/RESULT.json',result)
    original=dict(schema='SLA_DENSE_EXPERT_EPISODE_SHARDS_V3',state='committed',split='TRAIN',
        scene=row['parent'],scene_sha256=row['model_sha256'],source_sha=s['source_sha'],
        training_dataset_admitted=False,source_frame_count=321,source_command_count=640,label_count=61,
        source_proof=dict(encoded_windows=61,actual_two_second_success_tail=True,
            spec=pin(root/'EXECUTION_SPEC.json'),result=pin(root/'native-01/RESULT.json')))
    write(root/'data/sla-pt-shards/MANIFEST.private.json',original)
    return root,row,panel


def test_closed_episode_custody_matches_original_spec_result_and_fresh_panel(tmp_path):
    root,row,panel=source(tmp_path)
    assert source_episode(root,row,panel)[1]['label_count']==61
    # Even if someone edits the role/consumer in a copied spec, the source
    # recording's signed spec and the actual lease cannot silently follow it.
    p=root/'EXECUTION_SPEC.json';v=json.loads(p.read_text());v['task_learning_grasp']['prefix_atoms']=40;write(p,v)
    with pytest.raises(ValueError):source_episode(root,row,panel)


@pytest.mark.parametrize('mutation',['open','worker_failed','different_restart','different_scene','unconsumed'])
def test_running_failed_or_mismatched_episode_never_enters_cohort(tmp_path,mutation):
    root,row,panel=source(tmp_path)
    if mutation=='open':
        p=root/'control/LEASE.json';v=json.loads(p.read_text());v['status']='ACTIVE';write(p,v)
    elif mutation=='worker_failed':
        write(root/'WORKER_EXIT.private.json',dict(exit_code=1,failure='startup timeout'))
    elif mutation=='different_restart':row['restart']=6
    elif mutation=='different_scene':row['model_sha256']='f'*64
    else:
        p=root/'data/sla-pt-shards/MANIFEST.private.json';v=json.loads(p.read_text());v['source_command_count']=600;write(p,v)
    with pytest.raises(ValueError):source_episode(root,row,panel)
