"""One CPU contract check for endpoints, strata, deterministic order and cursor."""
from plan import schedule,strata

def check():
    assert [len(x) for x in strata(list(range(10)))]==[2,2,1,1,1,1,1,1]
    assert strata([1,2])==[[1],[2]]
    rows=[dict(seed=s,group=g,anchor=i,uid=f'{s}:{g}:{i}') for s in range(10000,10093) for g,n in enumerate([2,3,9,50]) for i in range(n)]
    a,groups,pools=schedule(rows);b,_,_=schedule(list(reversed(rows)));assert a==b and len(a)==8192
    for cursor in [0,8,16,1024,2048,4096,6144,8192]:
        assert a[:cursor]+a[cursor:]==a
        if cursor<8192:assert a[cursor]['unit']==cursor//2+1 and a[cursor]['micro']==0
    assert len(pools['F'])==16 and len(pools['B'])==77 and len(groups)==372
    print('PASS_CPU_COVER93_ENDPOINT_STRATA_ORDER_CURSOR_AND_POOL_CONTRACT')

if __name__=='__main__':check()
