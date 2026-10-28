"""Original296×j0/j1 evaluator bound to the actual newfit function globals."""
import argparse
from compat import load

def prepared():
    late=load('late');runtime=late.newfit.runtime
    # The wrapper exports aliases; assigning wrapper.evaluate does not update
    # the globals used by its exported main. Bind the underlying runtime once.
    runtime.evaluate=late.evaluate
    assert runtime.main.__globals__['evaluate'] is late.evaluate
    assert late.evaluate.__globals__['RAW']==runtime.RAW
    assert late.evaluate.__globals__['ROOT']==runtime.ROOT
    return runtime.main

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('EQUIL5120',),required=True);prepared()(p.parse_args().point)
