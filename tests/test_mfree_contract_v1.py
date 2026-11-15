from go2arx5_vla.world_model.mfree_contract_v1 import catalog


def test_finite_predeclared_populations_and_one_candidate():
    c=catalog(); train=c['train_attempts']; dev=c['dev16']
    assert len(train)==40 and len(dev)==16 and len(c['baseline8'])==8 and len(c['train8'])==8
    assert len({r['episode_id'] for r in train+dev})==56
    assert len([r for r in train if r['C4_member_if_success']])==4
    assert not {r['scene_lineage'] for r in train}&{r['scene_lineage'] for r in dev}
    assert not {tuple(r['xy_mm']) for r in train}&{tuple(r['xy_mm']) for r in dev}
    assert c['recipe']['updates']*c['recipe']['batch']==32000
    assert not c['recipe']['optional_successor_automatically_started']
