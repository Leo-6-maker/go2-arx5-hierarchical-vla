from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.sla_adapter.pd4_training_v71 import PD4Engine,fixed_probes,native_qualification


def cohort():
    # Complete stage/event coverage exercises the common row plan across two
    # distinct label identities without requiring a neural model in this test.
    phases=['PREGRASP']*3+['DESCEND']*18+['CLOSE']*5+['LIFT']*7+['HOLD']*28
    episodes=[dict(id=f'TRAIN{i:02d}-r14',task='PICK_HOLD',expert_stage_audit_only=phases)for i in (1,4,7,10)]
    rows=[(e,a)for e in episodes for a in range(61)]
    return SimpleNamespace(rows=rows,catalog_rows=lambda:[dict(split='TRAIN',task=e['task'],episode=e['id'],phase=e['expert_stage_audit_only'][a],anchor=a)for e,a in rows])


def test_views_share_visit_ids_and_distinct_probe_windows():
    a,b=PD4Engine.__new__(PD4Engine),PD4Engine.__new__(PD4Engine)
    x=a.visits_for_dataset(cohort());y=b.visits_for_dataset(cohort())
    assert x.shape==(16000,) and np.array_equal(x,y)
    assert len(a.selected_indices)==len(set(a.selected_indices))==16 and a.selected_indices==b.selected_indices
    assert set(x)==set(range(244))


def test_unqualified_native_statistics_never_admit_training(tmp_path):
    import json,hashlib
    f=tmp_path/'result';f.write_text(json.dumps(dict(schema='SLA_PD4_NEW_STATS_PROSPECTIVE_NATIVE_ACTUAL_V7_1',all_new_statistics_qualified=False)))
    with pytest.raises(ValueError,match='ALL_EIGHT'):native_qualification(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))


def test_PD4_package_supplies_codec_identity_without_changing_frozen_stats():
    from go2arx5_vla.sla_adapter.provider_pd4_v71 import Provider
    from go2arx5_vla.sla_adapter.provider_v3 import Provider as Original
    from go2arx5_vla.sla_adapter.pd4_training_v71 import OPERATION,STATISTICS
    stats=dict(fit_input='UNNORMALIZED_RELATIVE_XYZ_SO3_AND_LOGIT_GRIP',fit_unique_windows=244,split='TRAIN')
    p=Provider.__new__(Provider);pin=STATISTICS['PD4_FINAL_PLAN']
    p.manifest=dict(statistics_sha256=pin,bindings=dict(recipe=OPERATION,PD4_view='PD4_FINAL_PLAN',derived_statistics_sha256=pin,statistics_codec_metadata_source='PACKAGE_CODEC_VERSION_AND_EPSILON_WITH_FROZEN_PD4_QUANTILES'))
    p.validate_statistics_metadata(stats)
    with pytest.raises(ValueError,match='LOGIT_STATISTICS_VERSION_REQUIRED'):
        Original.__new__(Original).validate_statistics_metadata(stats)
    p.manifest['bindings']['PD4_view']='PD4_EXECUTED_REF'
    with pytest.raises(ValueError,match='EXACT_PD4'):p.validate_statistics_metadata(stats)
