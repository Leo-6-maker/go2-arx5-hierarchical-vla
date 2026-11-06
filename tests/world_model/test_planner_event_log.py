"""Exact legacy committed-line results, with bounded incremental decoding."""
import json
import os

import pytest

from go2arx5_vla.world_model import planner_event_log as module
from go2arx5_vla.world_model.target_reference import installed_target_identities


def legacy(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_committed_partial_blank_duplicate_and_expiry_evidence_match_legacy(tmp_path):
    path = tmp_path/'planner.jsonl'
    reader = module.PlannerEventLog(path)
    assert reader.read() == legacy(path)
    chunks = [b'{"kind":"accept","ref_id":"A"}', b'\n\n',
              b'{"kind":"lifecycle","reason":"PREFIX_TTL_', b'EXPIRED"}\n',
              b'{"kind":"accept","ref_id":"A"}\n', b'  \n',
              b'{"kind":"accept","ref_id":"B"}\n{"kind":']
    for chunk in chunks:
        with path.open('ab') as handle:
            handle.write(chunk)
        assert reader.read() == legacy(path)
        assert reader.read() == legacy(path)
    result = reader.read()
    result.clear()
    assert len(reader.read()) == 4


def test_malformed_committed_record_keeps_raising_until_actual_log_is_replaced(tmp_path):
    path = tmp_path/'planner.jsonl';reader = module.PlannerEventLog(path)
    path.write_bytes(b'{"kind":"accept"}\n');assert reader.read() == legacy(path)
    with path.open('ab') as handle:handle.write(b'{bad}\n')
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):reader.read()
        with pytest.raises(json.JSONDecodeError):legacy(path)
    replacement = tmp_path/'replacement';replacement.write_bytes(b'{"kind":"installed"}\n')
    os.replace(replacement,path)
    assert reader.read() == legacy(path)


def test_truncate_same_size_rewrite_delete_and_new_file_reset_history(tmp_path):
    path = tmp_path/'planner.jsonl';reader = module.PlannerEventLog(path)
    path.write_bytes(b'{"x":1}\n{"x":2}\n');assert reader.read() == legacy(path)
    path.write_bytes(b'{"x":3}\n');assert reader.read() == legacy(path)
    stamp=path.stat().st_mtime_ns
    path.write_bytes(b'{"x":4}\n');os.utime(path,ns=(stamp+1_000_000,stamp+1_000_000))
    assert reader.read() == legacy(path)
    path.unlink();assert reader.read() == []
    path.write_bytes(b'{"x":5}\n');assert reader.read() == legacy(path)


def test_decode_work_is_new_committed_records_and_original_trajectory_hashes_match(tmp_path,monkeypatch):
    path=tmp_path/'planner.jsonl';reader=module.PlannerEventLog(path);calls=[]
    original=module.json.loads
    monkeypatch.setattr(module.json,'loads',lambda line:(calls.append(len(line)),original(line))[1])
    row=dict(kind='installed',ref_id='A',times=[1.,2.],poses=[[1.,2.],[3.,4.]],inputs=[[0.],[0.]])
    for index in range(30):
        value=dict(row,monotonic_s=float(index))
        with path.open('ab')as handle:handle.write((json.dumps(value)+'\n').encode())
        count=len(calls);actual=reader.read();assert len(calls)-count==1
        for _ in range(3):
            count=len(calls);assert reader.read()==actual;assert len(calls)==count
        expected=legacy(path)
        assert actual==expected
        assert installed_target_identities(actual,'A')==installed_target_identities(expected,'A')


@pytest.mark.parametrize('phase', [
    'collect-target-fusion-task-phase-pose-continuation-v7',
    'collect-target-fusion-task-phase-bias-fade-v8'])
def test_only_new_explicit_phase_uses_incremental_reader_old_versions_keep_legacy(tmp_path,phase):
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    path=tmp_path/'planner.jsonl';path.write_text('{"kind":"accept","ref_id":"A"}\n')
    old=TargetSequentialAdapter({'phase':'collect-target-fusion-task-phase-pose-feedback-v6'},tmp_path)
    first=old.reference_events();first[0]['ref_id']='MUTATED_LOCAL_COPY'
    assert old.reference_events()[0]['ref_id']=='A' and not hasattr(old,'_planner_event_log')
    new=TargetSequentialAdapter({'phase':phase},tmp_path)
    assert new.reference_events()==legacy(path) and hasattr(new,'_planner_event_log')
    with path.open('ab')as f:f.write(b'{"kind":"lifecycle","reason":"PREFIX_TTL_EXPIRED"}\n')
    assert new.reference_events()==legacy(path)
