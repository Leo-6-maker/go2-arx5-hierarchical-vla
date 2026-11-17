import copy
import pytest
from go2arx5_vla.sla_adapter.shift7_cn38_r1_v4 import panel,evaluate


def bank():
    rows=[]
    for family,count in (('F1',14),('F2',4)):
        for i in range(count):
            for phase in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):
                rows.append(dict(index=len(rows),family=family,episode=f'{family}-{i:02d}',phase=phase))
    return dict(records=rows)


def predictions(n=80):
    cases=panel(bank())[:n]
    for row in cases:
        row.update(envelope=dict(original_full_H40_pass=True),original_compiler_pass=True,
            terminal_intent_matches=True,width=dict(expert_crossing_in_window=3 if row['phase']=='CLOSE'else None,
            crossing_error_s=0.,terminal_width_error_m=0.),coordination=dict(previous_actual_command_to_first_speed_m_s=0.),
            EEF_position_errors_m=[.001]*40,HYBRID_EEF_position_errors_m=[.001]*40)
    return cases


def test_locked_episode_order_first80_and_remaining100_without_repeat():
    cases=panel(bank());first=cases[:80]
    assert sorted({r['episode']for r in first if r['family']=='F1'})==['F1-01','F1-05','F1-08','F1-12']
    assert sum(r['family']=='F2'for r in first)==40
    assert sum(r['phase']=='CLOSE'for r in first)==16
    assert len({(r['entry'],r['noise'])for r in cases})==180
    assert sum(r['family']=='F1'for r in cases)==140
    assert sum(r['phase']=='CLOSE'for r in cases)==36


@pytest.mark.parametrize('n',[80,180])
def test_exact_terminal_sizes_and_fixed_global_GO(n):
    result=evaluate(predictions(n),expected=n)
    assert result['GO'] and result['predictions']==n
    with pytest.raises(ValueError):evaluate(predictions(n)[:-1],expected=n)


@pytest.mark.parametrize('fault',['suffix','event_missing','event_late','seam','intent','p95','max_point'])
def test_each_predeclared_failure_blocks_extension_without_tolerance_change(fault):
    rows=predictions();row=next(r for r in rows if r['phase']=='CLOSE')
    if fault=='suffix':row['envelope']['original_full_H40_pass']=False;row['original_compiler_pass']=False
    elif fault=='event_missing':row['width']['crossing_error_s']=None
    elif fault=='event_late':row['width']['crossing_error_s']=.15
    elif fault=='seam':row['coordination']['previous_actual_command_to_first_speed_m_s']=.04001
    elif fault=='intent':row['terminal_intent_matches']=False
    elif fault=='p95':
        for r in rows:r['EEF_position_errors_m']=[.00151]*40
    else:row['EEF_position_errors_m'][-1]=.00201
    before=copy.deepcopy(rows)
    assert not evaluate(rows,expected=80)['GO']
    assert rows==before
