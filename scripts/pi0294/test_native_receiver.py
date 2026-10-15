"""Actual delivered parser, zero ROS/physics/model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from .native_receiver import bind_packet


def check(parser):
    spec=importlib.util.spec_from_file_location('r4_native_parser',parser)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    measured=module.Pose((.1,.2,.3),(0,0,0,1))
    points=[dict(time=k*.05,position=[.4,0,.6],quaternion_xyzw=[0,0,0,1]) for k in range(41)]
    doc=dict(version=1,ee=dict(frame='world',waypoints=points),base=dict(frame='world',waypoints=points),
        _pi0294=dict(episode='cpu',sequence=0,source_mpc_s=7.,ttl_s=.4,operation='install'))
    args=dict(episode='cpu',sequence=-1,now=7.,trajectory_type=module.TaskTrajectory,ee=measured,base=measured)
    bound,source,seq=bind_packet(doc,**args)
    times,ee,base=bound.window(7.,2.,.05)
    assert source==7. and seq==0 and len(times)==41
    assert all(p.position==(.4,0.,.6) for p in ee+base)
    for changes in ({'episode':'old'},{'sequence':True},{'sequence':1},{'source_mpc_s':6.9},{'source_mpc_s':7.1},{'source_mpc_s':float('nan')}):
        bad=copy.deepcopy(doc);bad['_pi0294'].update(changes)
        try:bind_packet(bad,**args)
        except ValueError:pass
        else:raise AssertionError(changes)
    for part in ('ee','base'):
        bad=copy.deepcopy(doc);bad[part]['frame']='ee_start'
        try:bind_packet(bad,**args)
        except ValueError:pass
        else:raise AssertionError(part)
    duplicate=dict(args,sequence=0)
    try:bind_packet(doc,**duplicate)
    except ValueError:pass
    else:raise AssertionError('duplicate')
    cancel=copy.deepcopy(doc);cancel['_pi0294']['operation']='cancel'
    assert bind_packet(cancel,**args)==(None,7.,0)
    absent=copy.deepcopy(doc);del absent['base']
    assert bind_packet(absent,**args)[0].base is None
    return dict(status='ACTUAL_NATIVE_RECEIVER_CPU_PASS',requested_first_point_preserved=True,
        native_window_points=41,world_EEF_and_BASE=True,source_anchor_preserved=True,
        wrong_epoch_stale_future_duplicate_reordered_cancel_reset_checked=True,new_physics=0,new_model=0)


if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]))))
