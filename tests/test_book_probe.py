from copy import deepcopy
import pytest
from scripts.pi0294.multitask.book_probe import plan,validate_manifest


def test_small_diagnostic_preserves_train_split_noise_pairing_and_no_new_task_result():
    p=plan();assert len(p['windows'])==16 and len({(r['id'],r['anchor'])for r in p['windows']})==16
    assert {r['split']for r in p['windows']}=={'TRAIN'}and {r['id']for r in p['windows']}=={'BOOK_SOURCE_SEED000','BOOK_SOURCE_SEED004'}
    assert p['denoise_steps']==[4,16]and p['generation_queries']==32 and p['native_flow_loss_forwards']==16
    assert not p['DEV_or_TEST_used']and p['physics_integrations']==p['optimizer_updates']==p['formal_DEV_conditions']==0
    m=dict(checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],limits=dict(optimizer_updates=0,physics_integrations=0,generation_queries=32,model_loads=1))
    validate_manifest(m,p)
    bad=deepcopy(p);bad['windows'][0]['split']='DEV'
    with pytest.raises(PermissionError):validate_manifest(m,bad)
    bad=deepcopy(m);bad['limits']['optimizer_updates']=1
    with pytest.raises(PermissionError):validate_manifest(bad,p)
