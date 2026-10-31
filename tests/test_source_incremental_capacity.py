"""Exercise the actual cold-stage space guard with historical spent charges."""
from copy import deepcopy
from types import SimpleNamespace
import textwrap

import pytest
from scripts.pi0294.multitask.night_source_stage import PROGRAM


def capacity(free, historical):
    code=' storage='+PROGRAM.split(' storage=',1)[1].split('\nelse:\n budget=scope.forecast',1)[0]
    account={'charged_upper':{'Core_peak_bytes':historical}};before=deepcopy(account);receipts={}
    ns=dict(budget={'original_forecast':{'dimensions':{'Core_storage_bytes':{'projected':12000,'used':2000}}}},
            account=account,root='OWN_LOCAL_ROOT',raw=lambda x:x,
            os=SimpleNamespace(statvfs=lambda path:SimpleNamespace(f_bavail=free,f_frsize=1)),
            put=lambda name,value:receipts.update({str(name):value}))
    # The native source root is a Path; use a harmless path in this pure seam.
    from pathlib import Path
    ns['root']=Path('OWN_LOCAL_ROOT')
    exec(compile(textwrap.dedent(code),'actual-native-source-incremental-capacity','exec'),ns)
    assert account==before
    return next(iter(receipts.values()))


@pytest.mark.parametrize('historical',[0,20*1024**3,500*1024**3])
def test_spent_peak_charges_are_retained_but_not_charged_again_against_current_free(historical):
    required=10000+512*1024**2+64*1024**2+2*1024**3
    result=capacity(required,historical)
    assert result['historical_cumulative_peak_charged_bytes']==historical
    assert result['next_source_increment_upper_bytes']==576*1024**2
    assert result['retained_historical_files_already_excluded_from_free']is True
    assert result['old_future_bytes']==10000 and result['margin_bytes']==2*1024**3


def test_actual_next_write_old_protection_and_free_margin_are_still_required():
    required=10000+576*1024**2+2*1024**3
    with pytest.raises(AssertionError):capacity(required-1,0)
