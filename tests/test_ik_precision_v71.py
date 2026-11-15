import inspect
import pytest
from scripts.sla.diagnose_ik_precision_v71 import diagnostic_function,original


def test_isolated_clone_only_changes_declared_convergence_and_trace():
    before=inspect.getsource(original.certify_chunk)
    function,changes=diagnostic_function((20,1e-7,1e-6),lambda *args:None)
    assert changes==dict(loop=1,ep=2,er=2,trace=1)
    assert function.__globals__['contact_permitted'] is original.certify_chunk.__globals__['contact_permitted']
    for bound in (.7*(1+1e-10),1.*(1+1e-10),.05,.002,1e-10):
        assert bound in function.__code__.co_consts and bound in original.certify_chunk.__code__.co_consts
    assert inspect.getsource(original.certify_chunk)==before
    with pytest.raises(ValueError):diagnostic_function((1000,.1,.1),lambda *args:None)
