import pytest

from scripts.pi0292.durable import write
from scripts.sla.goal_v2_mix2000_eval import ARM, TRAINER, exact_result, pin


@pytest.mark.parametrize('step,status,code,expected', [
    (2000,'FIXED_SEGMENT_SAVED',0,(True,2000)),
    (1940,'BOUNDED_SAVED_PREFIX',75,(False,1940)),
    (2000,'BOUNDED_SAVED_PREFIX',75,None),
    (1940,'FIXED_SEGMENT_SAVED',0,None),
])
def test_only_exact_successful_milestone_releases_formal_eval(tmp_path, step, status, code, expected):
    train = tmp_path/'train'
    train.mkdir()
    result = train/'RESULT.private.json'
    write(result,dict(arm=ARM,source_sha=TRAINER,mode='train',global_step=step,status=status,
                      exit_code=code,new_native_requests=0,new_native_ticks=0))
    cold = dict(pid=0,cgroup_pids=[])
    if expected is None:
        with pytest.raises(ValueError,match='EXACT_SUCCESS_OR_VERIFIED_EARLY_PREFIX'):
            exact_result(tmp_path,cold,pin(result))
    else:
        assert exact_result(tmp_path,cold,pin(result)) == expected
    with pytest.raises(ValueError,match='ORIGINAL_COLD_TRAINING_RECEIPT'):
        exact_result(tmp_path,dict(pid=1,cgroup_pids=[1]),pin(result))
