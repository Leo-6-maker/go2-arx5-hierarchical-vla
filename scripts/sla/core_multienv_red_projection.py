"""Check the known F2 red pitch candidate against its new actual Wave A STOP."""
import argparse
import json
from pathlib import Path

import mujoco

from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.core_hopper_red_approach import PITCH_DEGREES, world_pitch
from scripts.sla.core_hopper_red_approach_audit import analyze_case
from scripts.sla.core_multienv_train import PRODUCER_SOURCE, PRODUCER_SPEC
from scripts.sla.hopper_parent_custody import digest

CONDITION = 'MEV1-TRAIN-F2-NEAR_NEG-00-HOLD-red'
ACTUAL = '1852d281138cdac76bf9bad070742dc5713e23646e78c8b04eb75e9df5c02b27'


def frozen_projection(root):
    if (digest(root/'EXECUTION_SPEC.json')!=PRODUCER_SPEC or mujoco.__version__!='3.2.7'
            or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the fixed Wave A geometry and no active controller')
    rows = json.loads((root/'RECORDS.private.json').read_bytes())
    row = next(r for r in rows if r['condition']==CONDITION)
    if (row['result']['sha256']!=ACTUAL or row['first_failure']!='ILLEGAL_CONTACT'
            or row['first_failure_signature']!=['ILLEGAL_CONTACT','DESCEND',[['task_blue_cube','x5_link6']]]):
        raise ValueError('The actual Wave A first contact differs from the registered inference')
    with forbid_integration(mujoco) as calls:
        result = analyze_case(root,row,(0,PITCH_DEGREES),world_pitch,source=PRODUCER_SOURCE)
        if calls:
            raise RuntimeError('Native integration was attempted in geometry readback')
    return dict(status='SAVED_TRAIN_GEOMETRY_ONLY',angles_deg=[0,PITCH_DEGREES],
                result=result,requests=0,integrations=0,model_generations=0,optimizer_updates=0,VAL_inputs=0)


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    print(json.dumps(frozen_projection(cli.parse_args().root)))
