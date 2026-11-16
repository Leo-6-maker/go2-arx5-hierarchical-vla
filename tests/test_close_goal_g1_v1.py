import numpy as np
from scripts.sla.close_goal_g1_v1 import choose_global, width_metrics, ROUTES


def test_global_selection_cannot_trade_nonclose_safety_for_close_gain():
    rows={r:dict(CLOSE_pass=0,nonCLOSE_pass=16)for r in ROUTES}
    rows['full_highest']=dict(CLOSE_pass=16,nonCLOSE_pass=15)
    assert choose_global(rows)=='cached_actual'
    rows['cached_highest']=dict(CLOSE_pass=8,nonCLOSE_pass=16)
    assert choose_global(rows)=='cached_highest'
    rows['full_actual']=dict(CLOSE_pass=8,nonCLOSE_pass=16)
    assert choose_global(rows)=='cached_highest'


def test_no_closure_does_not_win_crossing_or_terminal_width_metrics():
    target=np.zeros((40,10));target[:,9]=np.linspace(1,0,40)
    open_forever=target.copy();open_forever[:,9]=1
    report=width_metrics(open_forever,target)
    assert report['missed_in_window_crossing']
    assert report['crossing_error_s'] is None
    assert np.isclose(report['terminal_width_error_m'],.040)
    identical=width_metrics(target,target)
    assert identical['crossing_error_s']==0 and identical['terminal_width_error_m']==0
