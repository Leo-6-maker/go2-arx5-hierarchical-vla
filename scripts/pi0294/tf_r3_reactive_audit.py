"""Rebuild task references from saved native inputs, no solver or integration."""
import json
from pathlib import Path
import numpy as np
from .task_state import ActualState
from .tf_r3_source import sample
from .tf_r3_entry import retime
from .mp_client import Client
from .task_capture_runtime import read,sha
from .tf_r3_reactive import overhead,reference,VERSION
from scripts.pi0292.eval_contract import matrix10


def expected_windows(root,folder,model,packets,saved,intents):
    root,folder=Path(root),Path(folder);card=read(folder/'CARD.json');engine=ActualState(model)
    first=engine.update(packets[0]);first['time_s']=0.;initial=first['object_pose'].copy();selected=overhead(model,first,engine,card)
    binding=read(folder/'TASKFRAME_ENTRY_BINDING.json');assert binding['version']==VERSION
    assert np.allclose(selected['contact'],binding['overhead_world'],atol=1e-12,rtol=0)
    warm=root/'factory/requests'/(card['id']+'_WARMUP')/'0000';final=read(warm/'FINAL.private.json');route=read(warm/f"ROUTE-{final['route']:02d}.private.json")
    client=Client(root,engine);q=np.asarray(route['position']);entry=retime(dict(time=np.asarray(route['time'])*final['certificate']['uniform_time_scale'],poses=client.fk_path(first,q)),q)
    assert entry['entry_timing']==binding['entry_timing'];program=dict(time=entry['time'],poses=entry['poses'],gripper=np.full(len(entry['time']),.5))
    with np.load(folder/'TASKFRAME_ENTRY.npz',allow_pickle=False) as z:
        assert set(z.files)==set(program) and all(np.array_equal(program[k],z[k]) for k in program)
    expected={};mode={}
    for seq,intent in enumerate(intents):
        tick=seq*8;state=engine.update(packets[tick]);state['time_s']=saved[tick]['time_s'];at=float(state['time_s'])
        assert at==intent['persistent']['consumed_task_time_s']
        if card.get('reference_grid')=='PREFIX8_FULL040_END_SAMPLES_V2':
            persisted=intent['persistent']['reference_proof'];rejected=persisted['preinstallation_rate_rejections'];factors=(1.,1.25,1.5,2.)
            assert len(rejected)<4 and persisted.get('local_time_factor',1.)==factors[len(rejected)]
            accepted_index=intent['persistent']['planner_bundle_index'];assert accepted_index==seq*4+len(rejected)
            accepted_input=read(root/'factory/requests'/card['id']/f'{accepted_index:04d}'/'INPUT.private.json')
            for i,r in enumerate(rejected):
                assert r['attempt']==i and r['factor']==factors[i] and r['bundle_index']==seq*4+i
                rejected_folder=root/'factory/requests'/card['id']/f"{r['bundle_index']:04d}"
                failed=read(rejected_folder/'FINAL.private.json');request=read(rejected_folder/'INPUT.private.json')
                assert failed['input_sha256']==sha(rejected_folder/'INPUT.private.json') and failed['status']!='Success'
                assert failed['evidence']['routes']==r['routes'] and all(x['status'] in ('FROZEN_SOURCE_JOINT_ACCELERATION','FROZEN_SOURCE_JOINT_SPEED','FROZEN_SOURCE_RATE_REJECT') for x in r['routes'])
                for key in ('root_pose','joint_qpos','world_objects','actual_packet_arrays_sha256'):assert request[key]==accepted_input[key]
                assert [(x['IK_seed'],x['RRT_seed_table']) for x in request['goals']]==[(x['IK_seed'],x['RRT_seed_table']) for x in accepted_input['goals']]
        if at<entry['time'][-1]-1e-12:
            poses,_=sample(program,at+np.arange(40)*.05);stage='approach'
        else:
            previous=matrix10(np.asarray(intents[seq-1]['action_EEF13'])[7,:10]) if seq else entry['poses'][-1]
            pad=np.eye(4);g=engine.pad[0];pad[:3,:3]=engine.data.geom_xmat[g].reshape(3,3);pad[:3,3]=engine.data.geom_xpos[g]
            factor=intent['persistent']['reference_proof'].get('local_time_factor',1.)
            poses,stage,mode,proof=reference(state,previous,pad,card,initial,mode,local_time_factor=factor)
            persisted=intent['persistent']['reference_proof']
            for key in ('stage','desired_pad_world','actual_pad_world','reference_step_world','mode'):
                value=mode if key=='mode' else proof[key]
                assert value==persisted[key],key
        assert intent['stage']==stage if 'stage' in intent else True
        expected[seq]=poses
    return expected
