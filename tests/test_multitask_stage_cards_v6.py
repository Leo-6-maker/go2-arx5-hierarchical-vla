import pytest

from scripts.pi0292.durable import write
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases, teacher_slice


def test_partial_pilot_continues_only_unstarted_original_conditions():
    original = [dict(id=f'case{i}') for i in range(48)]
    population = dict(conditions=dict(TEACHER=original))
    prefix = [dict(row,task_success=False) for row in original[:3]]
    assert teacher_slice(population,prefix,6) == original[3:6]
    assert teacher_slice(population,original[:6],10) == original[6:10]
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,prefix,7)
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,[original[0],original[2]],6)
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,original[:6],49)


def test_registry_prior_scientific_failure_cannot_be_replaced(tmp_path):
    prior = tmp_path/'old'/'M-TEACHER-case'; (prior/'control').mkdir(parents=True)
    registry = tmp_path/'REGISTRY.json'
    write(registry,dict(entries=[dict(path=str(prior))]))
    write(prior/'control/LEASE.json',dict(status='CLOSED'))
    write(prior/'WORKER_EXIT.private.json',dict(not_started=True))
    require_unstarted_cases(registry,{prior.name})
    write(prior/'RESULT.private.json',dict(physical_requests=1,task_success=False))
    with pytest.raises(PermissionError,match='ALREADY_CONSUMED'):
        require_unstarted_cases(registry,{prior.name})
