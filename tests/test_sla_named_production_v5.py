import pytest

from go2arx5_vla.world_model import sla_named_expert_v5 as named
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_collection_v1 as collection
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import FROZEN_TEACHER
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id


def panel():
    return dict(schema=named.PRODUCTION_SCHEMA, task_learning_role=expert.ROLE,
        Teacher_implementation_source_sections_sha256=FROZEN_TEACHER,
        base_Teacher_qualification={'fixture':'existing'}, tasks=['PICK_HOLD'],
        D100_new=False, PICK_PLACE_admitted=False, old_engineering_episodes_reused=0,
        matched_actor_benchmark=False, other_live_collection_preserved=True,
        assignments=[dict(actor_count=3, layout=layout, restart=restart,
            prefix_atoms=10, parent=scene_id(layout), split='TRAIN', command=task.COMMAND,
            purpose=named.PRODUCTION_PURPOSE, Teacher_mode='NORMAL_TEACHER_V7',
            model_sha256='a'*64, native_run=f'fresh-{layout}-r{restart}')
            for restart in named.PRODUCTION_RESTARTS for layout in named.LAYOUTS])


def test_new_restarts_reuse_teacher_without_adopting_engineering_data(monkeypatch):
    calls=[]
    monkeypatch.setattr(collection,'qualified_base',lambda p:calls.append(p))
    p=panel()
    assert named.validate(p) == p['assignments']
    assert calls == [p['base_Teacher_qualification']]


@pytest.mark.parametrize('mutation',['old_restart','duplicate','D100','benchmark'])
def test_parallel_production_cannot_relabel_conditions_or_claim_isolated_throughput(monkeypatch,mutation):
    monkeypatch.setattr(collection,'qualified_base',lambda p:None)
    p=panel()
    if mutation=='old_restart':p['assignments'][0]['restart']=3
    elif mutation=='duplicate':p['assignments'][1]['native_run']=p['assignments'][0]['native_run']
    elif mutation=='D100':p['D100_new']=True
    else:p['matched_actor_benchmark']=True
    with pytest.raises(ValueError):named.validate(p)


def test_repaired_benchmark_is_fresh_and_fixed_at_all_three_actor_counts(monkeypatch):
    monkeypatch.setattr(collection,'qualified_base',lambda p:None)
    p=panel();p.update(schema=named.REPAIRED_BENCHMARK_SCHEMA,matched_actor_benchmark=True,
        prior_partial_benchmark_relabelled=False,startup_repair=dict(paused_readiness_s=600,
            total_wall_s=3600,original_native_markers_required=True,physics_or_Teacher_changed=False))
    p['assignments']=[dict(actor_count=a,layout=l,restart=r,prefix_atoms=10,parent=scene_id(l),
        split='TRAIN',command=task.COMMAND,purpose=named.REPAIRED_BENCHMARK_PURPOSE,
        Teacher_mode='NORMAL_TEACHER_V7',model_sha256='a'*64,native_run=f'bench-a{a}-{l}-r{r}')
        for a in named.ACTORS for l in named.LAYOUTS for r in named.REPAIRED_BENCHMARK_RESTARTS]
    assert len(named.validate(p))==48
    p['assignments'][0]['restart']=3
    with pytest.raises(ValueError):named.validate(p)
