import copy

from go2arx5_vla.world_model.task_learning_matched_v3 import summary,comparison


def rows(passes):
    result=[]
    for split,count in [('TRAIN',40),('DEV',20)]:
        for index in range(count):
            for noise in range(3):
                result.append(dict(split=split,phase=['PREGRASP','DESCEND','CLOSE','LIFT','HOLD'][index%5],
                    noise_group=noise,finite=True,envelope=dict(original_envelope_pass=index<passes,
                    metrics=dict(maximum_discrete_translation_acceleration_m_s2=.2 if index<passes else .5)),
                    errors=dict(pointwise=dict(XYZ_euclidean_m=[.0001]*40),
                        close_events=dict(missing=False,spurious=False),open_events=dict(missing=False,spurious=False)),
                    motion=dict(expert_displacement_m=.01,predicted_displacement_m=.01,direction_cosine=1.,displacement_ratio=1.)))
    return result


def test_full_noise_and_phase_denominators_keep_every_rejected_proposal():
    s=summary(rows(0))
    assert s['TRAIN']['overall']['requested']==120 and s['DEV']['overall']['requested']==60
    assert s['TRAIN']['overall']['envelope_pass']==0
    assert all(x['requested']==24 for x in s['TRAIN']['phases'].values())
    assert all(x['requested']==20 for x in s['DEV']['noises'].values())


def test_mechanism_gain_does_not_override_direction_collapse_or_nan():
    baseline=dict(summary=summary(rows(0)));candidate=dict(summary=summary(rows(40)))
    result=comparison(dict(A=baseline,C=candidate))
    assert result['D_admitted']is True
    collapsed=copy.deepcopy(candidate);collapsed['summary']['DEV']['overall']['median_displacement_ratio']=.01
    assert comparison(dict(A=baseline,C=collapsed))['D_admitted']is False
    bad=rows(40);bad[0]['finite']=False
    assert comparison(dict(A=baseline,C=dict(summary=summary(bad))))['D_admitted']is False
