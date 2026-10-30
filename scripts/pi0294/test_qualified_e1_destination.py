"""Actual pinned E1 plan and finite corrupted-stream checks; no HF or model IO."""
from copy import deepcopy
import hashlib,io,json,os
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_e1_destination as d


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('QUALIFIED_E1_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Exact existing private qualified E1 capture required')
    v=json.loads((Path(directory)/'QUALIFIED_E1_NATIVE01_COLD_SOURCE_REVISION_PREPARED.private.json').read_bytes())
    return dict(compiled_plan=v['cfg']['compiled_plan'],notices=v['cfg']['notices'],
        repo_id=d.REPO,revision=d.REVISION,prefix=d.PREFIX,limits=d.LIMITS)


def test_actual_already_qualified_E1_closed_evidence_and_fixed_fourteen_members(actual):
    files=d.expected_members(actual);proof=d.closed_source(actual)
    assert len(files)==14 and sum(r['bytes']for r in files.values())==12576408728
    assert proof['endpoint']['global_step']==512 and proof['endpoint']['cursor']==16384
    assert proof['endpoint']['official_inference_manifest_sha256']==d.CHECKPOINT
    assert proof['allocation_scientific_endpoint_complete']is False


@pytest.mark.parametrize('field,value',[('revision','0'*40),('prefix','other/model'),('repo_id','other/repo')])
def test_fixed_model_route_cannot_be_substituted(actual,field,value):
    v=deepcopy(actual);v[field]=value
    with pytest.raises(PermissionError):d.expected_members(v)


@pytest.mark.parametrize('corrupt',[False,True])
def test_stream_is_once_and_full_bytes_are_checked_before_publish(tmp_path,corrupt):
    path=tmp_path/'member';data=b'fixed model bytes';calls=[]
    expected=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    def source(url,timeout):calls.append((url,timeout));return io.BytesIO(b'x'*len(data)if corrupt else data)
    kwargs=dict(guard=lambda n:None,progress=lambda n:None,open_url=source)
    if corrupt:
        with pytest.raises(PermissionError):d.stream_one('fixture',path,expected,**kwargs)
        assert not path.exists()and path.with_name('member.inflight').is_file()
    else:
        assert d.stream_one('fixture',path,expected,**kwargs)==expected and path.read_bytes()==data
    with pytest.raises(PermissionError):d.stream_one('fixture',path,expected,**kwargs)
    assert len(calls)==1
