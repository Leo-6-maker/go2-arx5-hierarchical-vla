import base64,gzip,hashlib,time
from pathlib import Path
import pytest
from scripts.pi0294.multitask.night_source_stage import PROGRAM


def inputs(root):
    b=b'ACTUAL PREVIOUS SNAPSHOT';(root/'ORIGINAL_GLOBAL_CAPTURE.private.json').write_bytes(b)
    h=hashlib.sha256(b).hexdigest();now=time.time()
    return dict(predecessor_root=str(root),v7_finite_source_job={'already_validated':True},
        GLOBAL_capture_custody=dict(role='VERIFIED_CORE_CUSTODY_PROTECTED_FORECAST_NOT_CURRENT_GLOBAL_AUTHORITY',sha256=h,canonical_observed_epoch=now-3600),
        GLOBAL_gzip_base64=base64.b64encode(gzip.compress(b)).decode(),observation=dict(
            GLOBAL=dict(sha256=h,bytes=len(b),observed_epoch=now-3600,capture_is_fresh_native_authority=False),
            LOCAL=dict(observed_epoch=now)))


def check(p,policy=b'ALREADY_VALIDATED_EXACT_V7_BODY'):
    code="assert 0<=time.time()-P['observation']['LOCAL']['observed_epoch']<120"+PROGRAM.split("assert 0<=time.time()-P['observation']['LOCAL']['observed_epoch']<120",1)[1].split('root.mkdir(mode=0o700)',1)[0]
    exec(compile(code,'actual_source_custody_freshness','exec'),dict(P=p,time=time,Path=Path,
        v7_policy=policy,sha=lambda b:hashlib.sha256(b).hexdigest(),gzip=gzip,base64=base64))


def test_protected_custody_is_explicitly_stale_and_does_not_replace_fresh_Core(tmp_path):
    p=inputs(tmp_path);check(p)
    assert p['observation']['GLOBAL']['capture_is_fresh_native_authority']is False
    p['observation']['LOCAL']['observed_epoch']-=121
    with pytest.raises(AssertionError):check(p)


@pytest.mark.parametrize('fault',['no_finite','no_policy','fake_current','changed_bytes','changed_digest','false_time','no_role'])
def test_custody_cannot_fabricate_current_authority_or_use_an_unverified_snapshot(tmp_path,fault):
    p=inputs(tmp_path);policy=b'ALREADY_VALIDATED_EXACT_V7_BODY'
    if fault=='no_finite':p.pop('v7_finite_source_job')
    elif fault=='no_policy':policy=None
    elif fault=='fake_current':p['observation']['GLOBAL']['capture_is_fresh_native_authority']=True
    elif fault=='changed_bytes':(tmp_path/'ORIGINAL_GLOBAL_CAPTURE.private.json').write_bytes(b'CHANGED')
    elif fault=='changed_digest':p['GLOBAL_capture_custody']['sha256']='0'*64
    elif fault=='false_time':p['GLOBAL_capture_custody']['canonical_observed_epoch']+=1
    else:p['GLOBAL_capture_custody']['role']='CURRENT_GLOBAL'
    with pytest.raises(AssertionError):check(p,policy)
