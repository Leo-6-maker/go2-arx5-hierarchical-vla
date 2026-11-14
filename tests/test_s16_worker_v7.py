from types import SimpleNamespace
import numpy as np
import pytest

from scripts.sla.train_s16_offline_v7 import admission, prediction_metrics
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.codec_v1 import pad32
from go2arx5_vla.sla_adapter.s16_offline_v7 import OPERATION, DATA_SHA


def test_offline_worker_rejects_native_execution_or_changed_budget_before_loading_proofs():
    spec=dict(operation=OPERATION,resume=None,continuation=None,physics=False,
        scope='SLA_OFFLINE_S16_ONLY_V7',microbatch=8,training_matmul_precision='highest',
        dense_manifest_sha256=DATA_SHA,planned_candidate_updates=500,planned_target_visits=16000,
        planned_source_visits=0,closeout_reserve_s=600,native_execution_admitted=False)
    for change in [dict(physics=True),dict(native_execution_admitted=True),
                   dict(planned_candidate_updates=501),dict(resume={'path':'old-state'}),
                   dict(operation='N20_ANCHOR_EQUIVALENT_FM3K')]:
        with pytest.raises(ValueError,match='OFFLINE_ONLY'):
            admission(dict(spec,**change))


def test_s16_metrics_preserve_full_h40_limits_and_report_padding_separately():
    state=np.array([0.,0.,.7,1.,0.,0.,0.,1.,0.,1.])
    target=np.tile(state,(40,1))
    dataset=SimpleNamespace(rows=[({'id':'TRAIN01'},0)],labels={'TRAIN01':dict(
        nominal_policy_H40=target[None],latent7=np.zeros((1,40,7)),
        state_eef10=state[None],phase=['HOLD'])},statistics={'actions':Quantiles([-1.]*7,[1.]*7)})
    normalized=pad32(dataset.statistics['actions'].normalize(np.zeros((40,7)))).astype(np.float32)[None]
    before=prediction_metrics(dataset,0,normalized,target)
    assert before['EEF_full_H40_pass'] and before['generation_MSE7']==0
    normalized[:,:,7:]=2.
    physical=target.copy();physical[-1,0]=.01
    after=prediction_metrics(dataset,0,normalized,physical)
    assert not after['EEF_full_H40_pass']
    assert after['generation_MSE7']==0 and after['generation_MSE25']==4
    assert 'maximum_discrete_translation_acceleration_m_s2'in after['failed_metrics']
    assert after['joint_native_qualification']=='NOT_ESTABLISHED_BY_EEF_PROBES'
