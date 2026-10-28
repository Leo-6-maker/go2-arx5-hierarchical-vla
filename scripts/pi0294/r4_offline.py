"""Exact C0 native continuation; conditional C1 never bypasses its codec gate."""
import argparse
from pathlib import Path
from scripts.pi0292.durable import read
from .r4 import validate_entry
from .r3_offline import install_observer
from . import neural


def main():
    p=argparse.ArgumentParser()
    for n in ('root','request','bindings','resume'):p.add_argument('--'+n,required=True)
    a=p.parse_args();root=Path(a.root);validate_entry(root,profile='hopper-neural');request=read(a.request)
    if (request['mode']!='main' or request.get('arm')!='C0_V30_M4096' or request['resume']!=a.resume
            or request['stop_at']!=4096 or request['max_steps']!=4096 or request.get('workers')!=0):
        raise PermissionError('EXACT_C0_V30_M2048_CONTINUATION_ONLY')
    r0=read(root/'evidence/R0_MODEL_BEHAVIOR_GATE.json')
    if (r0['status']!='COMPLETE_ACTUAL_M2048_TRAIN4_DEV8_INPUT_CONTRACT_VALID'
            or r0['M2048_TRAIN_attempts']!=4 or r0['M2048_DEV_attempts']!=8):
        raise PermissionError('R4_EXISTING_MODEL_BEHAVIOR_MUST_PRECEDE_MORE_TRAINING')
    state=read(Path(a.resume)/'trainer_state.json');assert state['global_step']==2048
    a.mode='main';a.microbatch=2;a.workers=0;a.max_steps=4096;a.stop_at=4096
    install_observer(root,Path(a.resume),expected_step=2048)
    neural.run(a)


if __name__=='__main__':main()
