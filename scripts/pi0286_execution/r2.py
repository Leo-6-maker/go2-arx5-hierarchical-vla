"""PI0286 absolute phase guards; compatibility name for unchanged native callers."""
import json,time
from pathlib import Path
from contract import ROOT,guard
RELEASE='eae6dbc1f9281b204a4e28e5f7c100d705837318'
CANONICAL=6072813602
UPDATE_STOP=1791519000
MODEL_STOP=1791521700
ARM_START_STOP=1791516600
CORRECTION_START_STOP=1791515400
CONFIRM_START_STOP=1791518400

def publication(root=ROOT):
    assert Path(root).resolve()==ROOT.resolve()
    return guard()

def phase_allowed(kind,reserve=0,now=None):
    assert kind in ('update','model','physics') and reserve>=0
    assert not (ROOT/'runtime/SCIENTIFIC_STOP.json').exists(),'SCIENTIFIC_STOP'
    assert (time.time() if now is None else now)+reserve<(UPDATE_STOP if kind=='update' else MODEL_STOP),('PI0286_ABSOLUTE_PHASE_STOP',kind)

if __name__=='__main__':
    for kind,stop in [('update',UPDATE_STOP),('model',MODEL_STOP),('physics',MODEL_STOP)]:
        phase_allowed(kind,now=stop-1)
        try:phase_allowed(kind,now=stop)
        except AssertionError:pass
        else:raise AssertionError('Missing stop')
    print('PASS_PI0286_PHASE_BOUNDARIES_NO_MODEL_PHYSICS')
