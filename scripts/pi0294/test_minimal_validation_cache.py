"""Real canonical-byte keying, independent returns, finite memory, no failure memo."""
import pytest
from scripts.pi0294.generalization import minimal_validation_cache as m


def test_every_byte_and_generation_change_revalidates_and_mutating_return_cannot_poison():
    calls=[]
    @m.exact_ledger_validation
    def check(g,*,generation=None):
        calls.append((dict(g),generation))
        if g['status']!='CLOSED':raise PermissionError('ACTUAL_MUTATION_REJECTED')
        return dict(identity=dict(g),generation=generation)
    g={'status':'CLOSED','cost':1};a=check(g);a['identity']['cost']=999
    assert check(g)['identity']['cost']==1 and len(calls)==1
    g['cost']=2;assert check(g)['identity']['cost']==2 and len(calls)==2
    check(g,generation=1);assert len(calls)==3
    g['status']='ACTIVE'
    for _ in range(2):
        with pytest.raises(PermissionError):check(g)
    assert len(calls)==5


def test_bounded_eviction_and_no_nonfinite_key():
    calls=[]
    @m.exact_ledger_validation
    def check(g,*,generation=None):calls.append(g['value']);return g
    for x in range(m.MAX_ENTRIES+1):check({'value':x})
    check({'value':0});assert calls.count(0)==2
    with pytest.raises(ValueError):check({'value':float('nan')})
