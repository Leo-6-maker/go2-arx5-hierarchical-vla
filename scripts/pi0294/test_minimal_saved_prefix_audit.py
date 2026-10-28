"""CPU reader plan refusal contracts only; full GR00T state is native evidence."""
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import minimal_saved_prefix_audit as a


def fixture():
    return dict(schema=a.SCHEMA,scope_grant_sha256=a.GRANT,science_source_sha=a.SOURCE,
        science_manifest_sha256=a.MANIFEST,request_id=a.REQUEST,request_sha256=a.REQUEST_SHA,job_id=a.JOB,
        CPU_upper_s=480,walltime_upper_s=600,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0,
        formal_candidate=False,completed_session_reexecution=False,source_sha='a'*40,auditor_sha256='b'*64,
        metadata_root='/CPU_FIXTURE/new-metadata',science_root='/CPU_FIXTURE/closed-science',
        registry='/CPU_FIXTURE/registry.json',lease='/CPU_FIXTURE/new-own-lease.json',role='CPU_PLAN_FIXTURE',run_uuid='CPU_PLAN_FIXTURE',
        old_processes={k:dict(pid=i+1,procstart=1234+i)for i,k in enumerate(('owner','wrapper','prior_reader','prior_reader_wrapper'))})


def test_reader_plan_is_metadata_only_not_a_model_release():
    p=fixture();before=deepcopy(p);assert a.validate_plan(p)==before and p==before


@pytest.mark.parametrize('key,value',[('new_queries',1),('new_updates',1),('new_model_loads',1),('new_physics',1),
    ('CPU_upper_s',481),('walltime_upper_s',601),('source_sha',a.SOURCE),('request_id','OTHER'),
    ('completed_session_reexecution',True),('formal_candidate',True),('metadata_root','relative'),('old_processes',{})])
def test_reader_refuses_effects_replay_other_identity_or_larger_limits(key,value):
    p=fixture();p[key]=value
    with pytest.raises(PermissionError):a.validate_plan(p)
