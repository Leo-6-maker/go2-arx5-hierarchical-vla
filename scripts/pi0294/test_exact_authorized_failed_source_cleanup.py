"""Actual full before/after native metadata, not an invented cleanup fixture."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_portable_inventory as p

@pytest.fixture(scope='module')
def actual():
    root=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not root:pytest.skip('actual authorized native cleanup capture required')
    f=Path(root)/'MAINTENANCE30_CONTROL_IO/E2_COLD02_CLEANUP_BEFORE_AFTER_PROOF.private.json'
    packet=json.loads(f.read_bytes());body=gzip.decompress(base64.b64decode(packet['packet_gzip_base64']))
    assert __import__('hashlib').sha256(body).hexdigest()==packet['packet_sha256']
    return json.loads(body)

def test_actual_cleanup_and_original_unchanged_history(actual):
    old,new=actual['expected'],actual['actual'];oldbytes=json.dumps(old,sort_keys=True)
    assert p.verify(old,old)['portable_identity_sha256']==p.identity_digest(old['value'])
    proof=p.verify(old,new)
    assert proof['already_authorized_failed_source_cleanup']['removed_obsolete_source_files']==1493
    assert proof['already_authorized_failed_source_cleanup']['original_historical_gross_ceiling_and_costs_retained']
    assert proof['historical_allocated_bytes']==old['value']['allocated_bytes']
    assert proof['native_permission']is proof['ledger_changed']is False
    assert json.dumps(old,sort_keys=True)==oldbytes

@pytest.mark.parametrize('mutation',['file_bytes','inode','filesystem','missing_survivor','extra_file','directory_time','delete_record'])
def test_any_other_change_still_rejects(actual,mutation):
    new=deepcopy(actual['actual']);v=new['value'];row=next(r for r in v['rows']if r['kind']=='file')
    if mutation=='file_bytes':row['sha256']='0'*64
    elif mutation=='inode':row['inode']+=1
    elif mutation=='filesystem':next(iter(v['roots'].values()))['filesystem']['filesystem_id']+=1
    elif mutation=='missing_survivor':v['rows'].remove(row);v['allocated_bytes']-=row['allocated']
    elif mutation=='extra_file':r=deepcopy(row);r['path']+='-new';v['rows'].append(r);v['allocated_bytes']+=r['allocated']
    elif mutation=='directory_time':next(r for r in v['rows']if r['kind']=='directory')['mtime_ns']+=1
    else:next(r for r in v['rows']if r['path']=='FAILED_SOURCE_PAYLOAD_REMOVED_20261030.json')['sha256']='0'*64
    new['portable_sha256']=p.digest(v)
    with pytest.raises(PermissionError):p.verify(actual['expected'],new)
