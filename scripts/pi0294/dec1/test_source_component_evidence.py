"""Durable CPU-file fixtures, never native task or stop qualification."""
import json
import os
from unittest.mock import patch
import numpy as np
import pytest
from scripts.pi0294.dec1.source_component_evidence import SourceComponentEvidence

def fixture(tmp_path,**kwargs):
 return SourceComponentEvidence(tmp_path/'events.jsonl',episode_id='CPU',executor_id='SOURCE',maximum_bytes=kwargs.get('maximum',100000))


def test_actual_append_fsync_bytes_hash_chain_and_json_normalization(tmp_path):
 e=fixture(tmp_path);a=e.record(dict(kind='A',value=np.array([1.,2.])));b=e.record(dict(kind='B',episode_id='CPU'))
 rows=e.path.read_bytes().splitlines();assert len(rows)==2
 first,second=map(json.loads,rows);assert first['sha256']==a and second['sha256']==b and second['previous_sha256']==a
 assert e.read_events()==[dict(kind='A',value=[1.,2.]),dict(kind='B',episode_id='CPU')]
 e.close();assert len(e.read_events())==2
 with pytest.raises(PermissionError):e.record(dict(kind='C'))


def test_existing_file_is_never_overwritten(tmp_path):
 p=tmp_path/'events.jsonl';p.write_bytes(b'old')
 with pytest.raises(PermissionError):fixture(tmp_path)
 assert p.read_bytes()==b'old'

@pytest.mark.parametrize('bad',[dict(kind='A',episode_id='OTHER'),dict(kind='A',executor_id='OTHER'),dict(kind='')])
def test_bad_event_stays_latched_without_fabricated_append(tmp_path,bad):
 e=fixture(tmp_path)
 with pytest.raises((PermissionError,ValueError)):e.record(bad)
 assert e.path.read_bytes()==b'' and e.unknown
 with pytest.raises(PermissionError):e.record(dict(kind='A'))
 e.close()


def test_persisted_append_fsync_interruption_stays_unknown_no_retry(tmp_path):
 e=fixture(tmp_path)
 with patch('scripts.pi0294.dec1.source_component_evidence.os.fsync',side_effect=KeyboardInterrupt('FSYNC_REPLY_UNKNOWN')):
  with pytest.raises(KeyboardInterrupt):e.record(dict(kind='A'))
 actual=e.path.read_bytes();assert actual and e.sequence==0 and e.unknown
 with pytest.raises(PermissionError):e.record(dict(kind='A'))
 with pytest.raises(PermissionError):e.read_events()
 assert e.path.read_bytes()==actual;e.close()


def test_append_readback_failure_preserves_written_bytes_and_unknown(tmp_path):
 e=fixture(tmp_path)
 with patch.object(e,'_open_read',side_effect=OSError('READBACK_UNKNOWN')):
  with pytest.raises(OSError):e.record(dict(kind='A'))
 assert e.path.stat().st_size>0 and e.offset==0 and e.unknown;e.close()

@pytest.mark.parametrize('damage',['truncated','extra','changed','symlink'])
def test_independent_reader_rejects_partial_or_changed_file(tmp_path,damage):
 e=fixture(tmp_path);e.record(dict(kind='A'));raw=e.path.read_bytes()
 if damage=='truncated':e.path.write_bytes(raw[:-1])
 elif damage=='extra':e.path.write_bytes(raw+raw)
 elif damage=='changed':e.path.write_bytes(raw.replace(b'"A"',b'"B"'))
 else:
  p=tmp_path/'elsewhere';p.write_bytes(raw);e.path.unlink();e.path.symlink_to(p)
 with pytest.raises((ValueError,PermissionError,OSError)):e.read_events()
 assert e.unknown;e.close()


def test_normal_short_write_records_exact_bytes_without_replaying_any_prefix(tmp_path):
 e=fixture(tmp_path);original=os.write;calls=[]
 def short(fd,b):calls.append(len(b));return original(fd,b[:7])
 with patch('scripts.pi0294.dec1.source_component_evidence.os.write',side_effect=short):e.record(dict(kind='A'))
 assert len(calls)>1 and e.sequence==1 and e.read_events()==[dict(kind='A')];e.close()


def test_output_cap_checked_before_native_or_file_effect(tmp_path):
 e=fixture(tmp_path,maximum=10)
 with pytest.raises(PermissionError):e.record(dict(kind='A'))
 assert e.path.stat().st_size==0 and e.unknown;e.close()
