"""Fixed observation selection, without model generation or native qualification."""
from types import SimpleNamespace
import pytest
from go2arx5_vla.world_model.task_learning_probes_v1 import selections,PHASES
from go2arx5_vla.world_model.task_learning_export_v1 import PROFILE


def fixture():
    rows=[]
    stages=['PREGRASP']+['DESCEND']*7+['CLOSE','CLOSE']+['LIFT']*3+['HOLD']*2
    for recovery in(False,True):
        for layout in range(5):
            row=dict(id=f'episode-{layout}-{recovery}',parent=f'TRAIN{layout}',
                genuine_recovery_success=recovery,expert_stage_audit_only=stages.copy(),
                supervised_anchors=[a for a in range(15)if a!=8])
            # The optimizer's stage-balanced visits begin at a later phase and repeat.
            rows.extend((row,a)for a in [9,10,13]*3+[a for a in range(15)if a!=8])
    return SimpleNamespace(rows=rows,manifest=dict(profile=PROFILE),split='TRAIN')


def test_real_phase_rows_use_four_layouts_and_never_count_repeated_visits():
    data=fixture();result=selections(data)
    assert len(result)==40 and len({i for i,_ in result})==40
    assert [p for _,p in result]==list(PHASES)*8
    assert [data.rows[i][1]for i,_ in result]==[0,1,9,10,13]*8
    assert {data.rows[i][0]['parent']for i,_ in result}=={f'TRAIN{i}'for i in range(4)}
    assert sum(data.rows[i][0]['genuine_recovery_success']for i,_ in result)==20


def test_dev_uses_only_four_fixed_reference_episodes():
    data=fixture();data.split='DEV'
    result=selections(data)
    assert len(result)==20
    assert all(not data.rows[i][0]['genuine_recovery_success']for i,_ in result)


def test_absent_phase_is_not_filled_with_a_hold_or_masked_intervention():
    data=fixture()
    for row,anchor in data.rows:
        row['supervised_anchors']=[a for a in row['supervised_anchors']if a!=9]
    with pytest.raises(ValueError,match='REAL_SUPERVISED_DECISION_REQUIRED_FOR_PHASE_CLOSE'):
        selections(data)


def test_old_short_probes_keep_the_original_first_four_episode_queries():
    data=fixture();data.manifest['profile']='TARGET_GR00T_V1'
    result=selections(data)
    assert len(result)==4 and all(p is None for _,p in result)
    assert [data.rows[i][1]for i,_ in result]==[9]*4


def test_multiple_real_phase_artifacts_survive_package_reconstruction(tmp_path):
    import json
    import numpy as np
    from go2arx5_vla.world_model.task_learning_probes_v1 import contract,artifact
    from go2arx5_vla.world_model.target_gr00t_delivery_v1 import compare_generations
    data=fixture();records=[]
    for index,phase in selections(data):
        row,anchor=data.rows[index]
        records.append(dict(episode=row['id'],split='TRAIN',anchor=anchor,seed=len(records),
            noise=dict(fixture=len(records)),phase_audit_only=phase,
            genuine_recovery_audit_only=row['genuine_recovery_success'],
            sample_file=row['id']+f'-a{anchor:02d}.npz'))
    for name in('first','second'):
        folder=tmp_path/name;folder.mkdir()
        for record in records:
            np.savez(artifact(folder,record),physical_H40=np.zeros((40,10)),normalized=np.zeros((1,40,132)))
        (folder/'RESULT.json').write_text(json.dumps(dict(records=records,selection_contract=contract())))
    a,b=tmp_path/'first',tmp_path/'second'
    assert compare_generations(a,b)['actual_matched_queries']==40
    meta=json.loads((b/'RESULT.json').read_text());meta['records'].pop()
    (b/'RESULT.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='SAME_FIXED_REAL_PHASE_SELECTION_CONTRACT'):
        compare_generations(a,b)
