"""Real temp inode mutations; identities/budgets are explicit CPU fixtures."""
import hashlib,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_core_d25_output_guard as g,mix_core_d25_local_successor as s,mix_accounting as a

@pytest.fixture
def guarded(tmp_path,monkeypatch):
    parent=tmp_path/'campaign';neural=parent/'neural-children'/s.RUN;physical=parent/'children'/s.RUN;owner=parent/'control/owner03'
    for p in (neural,physical,owner):p.mkdir(parents=True)
    old=parent/'control/closed02';old.mkdir();(old/'metadata.json').write_text('{}')
    delivery=parent/'control/delivery';model=delivery/'model_view';model.mkdir(parents=True)
    uid=os.getuid()
    # Production requires Core uid2009. Only the standalone fixture reader's
    # uid observation is replaced; all lstat/inode/mtime/bytes are real.
    original=g.inventory
    def inventory(path,*,uid,maximum=100000):return original(path,uid=os.getuid(),maximum=maximum)
    monkeypatch.setattr(g,'inventory',inventory)
    for name in s.NINE:
        p=model/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPU_FIXTURE');p.chmod(0o400)
    rows={}
    for p in (old,delivery):rows.update(original(p,uid=uid)[0])
    sha=hashlib.sha256((json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    marker=dict(replacement_request_id='future-rid',physical_inventory_join=dict(model_root=str(model),
        model_stats={n:dict(stat=rows[str(model/n)])for n in s.NINE},Core_inventory_sha256=sha,
        roots=dict(old=dict(path=str(old)),delivery=dict(path=str(delivery)))),paid_IO_credit=dict(Hopper_storage_bytes=1024**2,Core_storage_bytes=1024**2))
    path=neural/g.BINDING;path.parent.mkdir();path.write_bytes(a.raw(marker));path.chmod(0o400)
    lb=neural/g.LOCAL_BINDING;lb.write_text(json.dumps(dict(owner_root=str(owner))))
    manifest=dict(execution=dict(sealed_inputs={g.BINDING:s.digest(marker)}))
    monkeypatch.setattr(s,'validated',lambda local:marker)
    # Match the production uid fence without replacing inode observations.
    guard=g.Guard.__new__(g.Guard);guard.root=neural;guard.clock=lambda:0;guard.last=None;guard.proof=None
    guard.marker=marker;guard.join=marker['physical_inventory_join'];guard.owner=owner;guard.physical=physical;guard.uid=uid
    (neural/'model_views').mkdir();(neural/'model_views/D25').symlink_to(model,target_is_directory=True)
    return guard,old,model

def test_true_temp_inodes_and_paid_two_bucket_outputs(guarded):
    guard,_,_=guarded;v=guard.check(force=True)
    assert v['model_inode_members']==9 and v['new_tensor_copies']==v['model_streams']==0 and not v['native_permission']

@pytest.mark.parametrize('change',['old_metadata','model_inode','new_tensor','new_alias','overflow'])
def test_changed_closed_payload_duplicate_model_alias_and_overflow_stop(guarded,change):
    guard,old,model=guarded
    if change=='old_metadata':(old/'metadata.json').write_text('{"changed":true}')
    elif change=='model_inode':
        p=model/s.NINE[0];p.chmod(0o600);p.write_bytes(b'CHANGED')
    elif change=='new_tensor':(guard.root/'new.safetensors').write_bytes(b'COPY')
    elif change=='new_alias':(guard.physical/'outside').symlink_to('/tmp',target_is_directory=True)
    else:
        guard.marker=deepcopy(guard.marker);guard.marker['paid_IO_credit']['Core_storage_bytes']=1
    with pytest.raises(PermissionError):guard.check(force=True)
