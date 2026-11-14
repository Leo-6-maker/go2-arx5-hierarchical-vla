import copy
import pytest
from go2arx5_vla.sla_adapter import reference_dataset_v6 as ref
from go2arx5_vla.sla_adapter.named_dataset_v5 import CONDITIONS
from go2arx5_vla.sla_adapter.gripper_v2 import VERSION as CODEC


def manifest():
    return dict(schema=ref.SCHEMA,state='committed',dataset_id=ref.DATASET_ID,scope=ref.SCOPE,
        tasks=['PICK_HOLD'],objects=1,split='TRAIN',codec=CODEC,D100_new=False,D300_new=False,
        PICK_PLACE_admitted=False,legacy_engineering_receipts_modified=False,old_2EP_reused=0,
        old_engineering16_reused=0,phase_input_to_student=False,consumption_proven=True,
        actual_success_tail_s=2.,command_period_ticks=25,frame_period_ticks=50,anchor_period_ticks=250,
        physics_dt_s=.002,supervised_windows=1220,
        selection_rule='FIRST20_DECLARED_ASSIGNMENTS_ALL_COMPLETE_NO_SUBSTITUTION',
        episodes=[dict(id=f'{l}-{r}',layout=l,restart=r,split='TRAIN',task='PICK_HOLD',prefix_atoms=10,
            windows=61,frames=321,commands=640,native_integrations=21703)for l,r in CONDITIONS],
        label_semantics=ref.VERSION,original_target_manifest=dict(sha256=ref.ORIGINAL_MANIFEST_SHA),
        original_observations_and_states_unchanged=True,all40_target_knots_actually_consumed=True,
        native_reference_transport_changed=False)


def test_new_label_identity_keeps_the_complete_original_population_and_clock():
    m=manifest();ref.manifest_contract(m)
    for field,value in [('dataset_id','old-V5'),('label_semantics','issuing-query-nominal'),
            ('original_observations_and_states_unchanged',1),('all40_target_knots_actually_consumed',False),
            ('native_reference_transport_changed',True),('physics_dt_s',.004),('D100_new',True)]:
        bad=copy.deepcopy(m);bad[field]=value
        with pytest.raises(ValueError):ref.manifest_contract(bad)
    bad=copy.deepcopy(m);bad['episodes'].pop()
    with pytest.raises(ValueError):ref.manifest_contract(bad)


def test_new_training_cannot_be_an_extra_segment_of_old_representation(monkeypatch):
    from scripts.sla.train_reference_target_v6 import admission
    s=dict(operation='OLD_N20',resume=None,continuation=None,reference_label_semantics=ref.VERSION,
        microbatch=8,training_matmul_precision='highest',closeout_reserve_s=120,
        planned_candidate_updates=3000,planned_target_visits=96000,planned_source_visits=0)
    for change in [dict(),dict(operation=ref.OPERATION,resume={'path':'old500'}),
            dict(operation=ref.OPERATION,planned_candidate_updates=6000)]:
        with pytest.raises(ValueError,match='NEW_REFERENCE_LABEL_CLEAN_BASE_FIXED3K'):
            admission(dict(s,**change))
