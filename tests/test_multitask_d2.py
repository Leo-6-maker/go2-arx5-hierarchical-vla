import numpy as np

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import rotation_increment
from go2arx5_vla.world_model.multitask_placement_geometry_d2 import landing_tcp, remaining_placement, transform
from scripts.wm02.multitask_d2_campaign import matrix_still_possible


def test_touchdown_uses_measured_se3_and_rotated_support_footprint():
    attachment = transform(rotation_increment(np.array([.1,.2,0])), [.04,0,-.015])
    rotation = rotation_increment(np.array([.1,-.22,-.35]))
    half = np.array([.02,.025,.021])
    tcp = landing_tcp(attachment,[.4,.12],rotation,half,.61)
    lowered = tcp@attachment
    assert np.allclose(lowered[:2,3],[.4,.12])
    assert np.isclose(lowered[2,3]-abs(lowered[2,:3])@half,.61)
    initial=tcp.copy()
    initial[2,3]+=.04
    knots=remaining_placement(initial,initial@attachment,tcp,np.full(2,.020),np.zeros(2),
                              supported=False,withdrawal=[0,0,.06])
    assert np.allclose(knots[1].object_pose,lowered)
    assert np.allclose(knots[3].object_pose,lowered)
    assert np.isclose(np.linalg.norm(knots[3].tcp[:3,3]-tcp[:3,3]),.06)


def test_futility_keeps_in_flight_and_excludes_non_g7_success():
    cards = [dict(panel_entry_id=str(k),mobile_condition={'family':'F1'},
                  task_instruction_id='PLACE-red-left') for k in range(4)]
    complete = {'task_and_mobile_success':True,'G7_pass':True}
    rejected = {'task_and_mobile_success':True,'G7_pass':False}
    assert matrix_still_possible(cards,{'0':rejected,'1':rejected})
    assert not matrix_still_possible(cards,{'0':rejected,'1':rejected,'2':rejected})
    assert matrix_still_possible(cards,{'0':rejected,'1':rejected,'2':complete})


def test_world_queue_separate_bounded_requests_and_round_robin(tmp_path):
    import pytest

    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.hybrid_nonphysical_v3 import identity
    from go2arx5_vla.world_model.multitask_queue_d2 import WorldQueue

    entries = {world:[str(tmp_path/'cases'/world)] for world in ('W1','W2')}
    queue = WorldQueue(tmp_path,entries)

    def query(world,n):
        capture = tmp_path/'cases'/world/f'q{n}'
        (capture/'inference-control').mkdir(parents=True)
        (capture/'model-response').mkdir()
        live = capture/'inference-control'/'live.json'
        write(live,dict(cancelled=False,stop=False))
        request = dict(input={'path':str(capture/'INPUT.npz'),'sha256':'a'*64},
                       clock_mode='SIM_PAUSED_ONLY',episode=f'{world}-q{n}',native_step=5703+n*250,
                       bindings={'policy_package_sha256':'b'*64})
        request['request_id']=identity(request)
        return dict(request=request, response_root=str(capture/'model-response'),
                    live_path=str(live), branch_barriers=False)

    a,b,c = query('W1',0),query('W2',0),query('W1',1)
    route = queue.publish('W1',a)
    assert queue.publish('W1',a)==route
    with pytest.raises(ValueError,match='OUTSIDE_OWNED_EPISODE'):
        queue.publish('W2',a)
    with pytest.raises(RuntimeError,match='ALREADY_HAS'):
        queue.publish('W1',c)
    queue.publish('W2',b)
    one = queue.next(set())
    two = queue.next({one['request']['request_id']})
    assert (one['d2_route']['environment_id'],two['d2_route']['environment_id'])==('W1','W2')
    write(a['live_path'],dict(cancelled=True,stop=False))
    queue.publish('W1',c)
    assert queue.next({route['request_id'],two['request']['request_id']})['d2_route']['native_step']==5953
