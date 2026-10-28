"""No native calls: preserve strict incremental bounds and unrelated bytes."""
import math
from types import SimpleNamespace
import pytest
from . import administrative_cpu_tail as unit


@pytest.mark.parametrize('used',(0,.9,7.99,8.765,12.01,1790))
def test_includes_prior_process_cpu_with_no_extra_tail_or_paid_upper(used):
    seen=[]
    resource=SimpleNamespace(RLIMIT_CPU=0,setrlimit=lambda kind,limits:seen.append(limits))
    time=SimpleNamespace(process_time=lambda:used)
    code='began=time.process_time();'+unit.OLD+'\nresult="same financial proposal"\n'
    changed=unit.bind(code)
    namespace=dict(time=time,resource=resource)
    exec(compile(changed,'<CPU-only-limit-fixture>','exec'),namespace)
    soft,hard=seen[0]
    assert soft==math.floor(used)+8 and hard==math.floor(used)+10
    assert 0<soft-used<=8 and 0<hard-used<=10 and hard<=1800
    assert namespace['result']=='same financial proposal'
    assert changed.replace(unit.NEW,unit.OLD)==code


@pytest.mark.parametrize('program',('',unit.OLD,'began=time.process_time();'+unit.OLD+';'+unit.OLD,
    unit.OLD+';began=time.process_time();'))
def test_only_exact_known_boundary_can_be_transformed(program):
    with pytest.raises(PermissionError):unit.bind(program)


def test_paid_process_upper_rejects_before_limit_or_financial_body():
    code=unit.bind('began=time.process_time();'+unit.OLD+'\nmutate()')
    called=[]
    with pytest.raises(AssertionError):
        exec(code,dict(time=SimpleNamespace(process_time=lambda:1790.1),
            resource=SimpleNamespace(RLIMIT_CPU=0,setrlimit=lambda *a:called.append(a)),
            mutate=lambda:called.append('forbidden')))
    assert called==[]
