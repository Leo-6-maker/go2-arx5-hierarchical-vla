"""Route-selection fixtures, never scan/restore or native qualifications."""
import copy
import pytest
from go2arx5_vla.world_model import task_learning_scan_route_v2 as route


def queued(state='Q'):
    return dict(PBS=dict(job_state=state), lease=dict(status='PBS_SUBMITTED'),
                worker=None, result=None)


def closed():
    return dict(PBS=dict(job_state='F', Exit_status=0),
                lease=dict(status='CLOSED', worker_exit_code=0),
                worker=dict(exit_code=0), result=dict(status='fixture-only'))


@pytest.mark.parametrize('state', ['Q', 'H'])
def test_only_unstarted_waiting_scan_uses_independent_micro1(state):
    assert route.choose(queued(state)) == route.USE_QUEUE_MICRO1


@pytest.mark.parametrize('lease', ['PBS_SUBMITTED', 'ACTIVE', 'CLOSED'])
def test_running_scan_keeps_its_existing_selected_batch_path(lease):
    value = queued('R'); value['lease']['status'] = lease
    original = copy.deepcopy(value)
    assert route.choose(value) == route.WAIT_RUNNING_SCAN
    assert value == original


def test_finished_scan_requires_actual_worker_and_lease_before_preparer():
    assert route.choose(closed()) == route.USE_SCAN_PATH
    for field in ['PBS', 'lease', 'worker', 'result']:
        value = closed(); value[field] = None
        with pytest.raises(ValueError): route.choose(value)
    for field, key in [('PBS', 'Exit_status'), ('lease', 'worker_exit_code'), ('worker', 'exit_code')]:
        value = closed(); value[field][key] = 1
        with pytest.raises(ValueError): route.choose(value)


def test_stale_queue_metadata_or_unknown_state_never_authorizes_a_successor():
    for changes in [dict(lease=dict(status='ACTIVE')), dict(worker=dict(exit_code=0)),
                    dict(result=dict(status='fixture')), dict(PBS=dict(job_state='S')),
                    dict(PBS={})]:
        with pytest.raises(ValueError): route.choose(dict(queued(), **changes))
