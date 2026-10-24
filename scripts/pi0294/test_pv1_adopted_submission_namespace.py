"""CPU namespace tests; no scheduler, model or native qualification."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.pi0294.pv1 import resources
from scripts.pi0294.pv1 import pre_model_qualification as qualification
from scripts.pi0294.pv1 import transport_fixture as transport
from scripts.pi0294.pv1.decision import AUTHORITY
from scripts.pi0294.dec1 import b_heartbeat_recovery
from scripts.pi0294.native_dataset import sha


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));path.chmod(0o400)


@pytest.fixture
def root(tmp_path):
    root=tmp_path/'campaign'/'children'/'evaluation'
    root.mkdir(parents=True)
    return root


def ledger(root):
    value=dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',
        authority_release=AUTHORITY,campaign_run_uuid='campaign')
    put(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json',value)
    return value


@pytest.mark.parametrize('number',[1,12,24])
def test_original_namespaces_keep_original_preparation_semantics(root,number):
    assert resources.namespace_submission(root,number)==number


@pytest.mark.parametrize('number',[0,-1,True,25,28,100])
def test_no_missing_ledger_or_arbitrary_ordinal_extension(root,number):
    with pytest.raises(PermissionError):resources.namespace_submission(root,number)


def test_an_unadopted_limit_field_or_partial_marker_cannot_extend_authority(root):
    value=ledger(root);value['global_PBS_limit']=1000
    p=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
    for changed in (value,dict(value,**{b_heartbeat_recovery.FIELD:dict(grant_sha256='0'*64)})):
        p.chmod(0o600);put(p,changed)
        with pytest.raises(PermissionError):resources.namespace_submission(root,25)


def test_both_nominal_and_isolated_fault_namespaces_consume_actual_parent_cap(root):
    value=ledger(root)
    with patch.object(resources,'pbs_limit',return_value=28) as cap:
        for prefix in (transport.BASE,transport.BASE+'-FAULT-LATE',transport.BASE+'-FAULT-EXIT'):
            for number in (25,28):
                folder,n,mode=transport.namespace(root,prefix+'-ALLOCATION-'+str(number))
                assert n==number and mode==prefix and folder.is_relative_to(root)
        assert all(c.args==(value,) for c in cap.call_args_list)
        with pytest.raises(PermissionError):transport.namespace(root,transport.BASE+'-ALLOCATION-29')


@pytest.mark.parametrize('field',['schema','authority_release','campaign_run_uuid'])
def test_foreign_parent_ledger_rejected_before_any_cap_lookup(root,field):
    value=ledger(root);value[field]='foreign'
    p=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';p.chmod(0o600);put(p,value)
    with patch.object(resources,'pbs_limit') as cap,pytest.raises(PermissionError):
        resources.namespace_submission(root,28)
    cap.assert_not_called()


def test_parent_ledger_symlink_never_authorizes_a_namespace(root,tmp_path):
    outside=tmp_path/'outside.json';put(outside,dict(global_PBS_limit=1000))
    (root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').symlink_to(outside)
    with pytest.raises(PermissionError):resources.namespace_submission(root,28)


def test_current_admission_above_24_still_needs_matching_immutable_request_and_live_job(root):
    ledger(root);directory=root/'pbs/alloc-28'
    request=dict(submit_count=28)
    put(directory/'REQUEST.json',request)
    allocation=dict(status='PASS_PV1_OWN_NATIVE_H200_ALLOCATION',job_id='fixture.hopper-m-02',
        hostname='fixture-h200',allocation_start_epoch=1000,allocation_end_epoch=10000,
        request_sha256=sha(directory/'REQUEST.json'))
    put(directory/'ALLOCATION_ADMISSION.json',allocation)
    with patch.object(resources,'pbs_limit',return_value=28),\
            patch.dict(qualification.os.environ,{'PBS_JOBID':allocation['job_id']}),\
            patch.object(qualification.socket,'gethostname',return_value=allocation['hostname']),\
            patch.object(qualification.time,'time',return_value=2000):
        assert qualification.current_directory(root,allocation)==directory
        # Merely extending the namespace cannot relabel an A40 as H200.
        with pytest.raises(PermissionError):
            qualification.current_directory(root,dict(allocation,status='PASS_DEC1_OWN_NATIVE_A40_SOURCE_TRAIN_ALLOCATION'))
        p=directory/'REQUEST.json';p.chmod(0o600);put(p,dict(submit_count=27))
        with pytest.raises(PermissionError,match='NAMESPACE_SUBMISSION'):
            qualification.current_directory(root,allocation)
        p.chmod(0o600);put(p,request)
        put(directory/'PBS_PROCESS_FINAL.json',{})
        with pytest.raises(PermissionError,match='TERMINAL_ALLOCATION'):
            qualification.current_directory(root,allocation)
