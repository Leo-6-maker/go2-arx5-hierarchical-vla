import hashlib
from io import BytesIO

import pytest
from scripts.pi0294.multitask import raw_parent_delivery as delivery


def pin(body):return dict(sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))


def test_only_the_original_seven_raw_files_and_finite_input_scope():
    from scripts.pi0294.generalization.minimal_model import FILES
    from scripts.pi0294.generalization.minimal_training import RAW_REVISION,RAW_PARENT_FILES_ID
    assert {n:v[0]for n,v in delivery.FILES.items()}==FILES
    assert delivery.REVISION==RAW_REVISION and delivery.PARENT_ID==RAW_PARENT_FILES_ID
    s=delivery.spec('/var/cache/users/liuyu-nfs/new-raw',100,now=500.)
    assert s['total_input_bytes']==6914922850 and s['deadline_epoch']==4100.
    assert s['neural_models_GPU_physics_optimizer_updates']==0 and s['training_admitted']is False
    assert s['maximum_bytes_per_second']==8*1024**2 and s['free_margin_bytes']==30*1024**3


def test_streamed_exact_input_has_complete_stat_bound_audit(tmp_path):
    body=b'fixed raw bytes'*1000;observed=[];path=tmp_path/'part.safetensors'
    row=delivery.download_member(BytesIO(body),path,pin(body),guard=lambda:None,
        report=observed.append,throttle=lambda n:None)
    assert path.read_bytes()==body and not path.with_suffix('.safetensors.PARTIAL').exists()
    assert row==dict(sha256=pin(body)['sha256'],stat=delivery.stat_identity(path))
    assert observed[-1]==len(body)


@pytest.mark.parametrize('fault',['short','corrupt','oversize'])
def test_incomplete_or_corrupt_transfer_stays_partial_not_qualified(tmp_path,fault):
    original=b'correct';bad={'short':b'corr','corrupt':b'badxxxx','oversize':b'correct-more'}[fault]
    observed=[];path=tmp_path/'weights'
    with pytest.raises(PermissionError):
        delivery.download_member(BytesIO(bad),path,pin(original),guard=lambda:None,
            report=observed.append,throttle=lambda n:None)
    assert not path.exists()and(tmp_path/'weights.PARTIAL').exists()
    assert observed[-1]==len(bad)  # Failed overlong HTTP bytes are still charged.


def test_existing_input_or_unknown_partial_is_never_replaced(tmp_path):
    path=tmp_path/'weights';body=b'protected';path.write_bytes(body)
    with pytest.raises(PermissionError):
        delivery.download_member(BytesIO(b'new'),path,pin(b'new'),guard=lambda:None,report=lambda n:None,throttle=lambda n:None)
    assert path.read_bytes()==body
    path2=tmp_path/'new';partial=tmp_path/'new.PARTIAL';partial.write_bytes(b'previous-unknown')
    with pytest.raises(FileExistsError):
        delivery.download_member(BytesIO(b'new'),path2,pin(b'new'),guard=lambda:None,report=lambda n:None,throttle=lambda n:None)
    assert partial.read_bytes()==b'previous-unknown'and not path2.exists()
