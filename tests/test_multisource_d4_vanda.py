import hashlib
import json

import pytest

from go2arx5_vla.sla_adapter.multisource_d4 import locate_input
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.wm02.multisource_d4_vanda import allocation_window


def test_archived_input_keeps_original_digest_after_relocation(tmp_path):
    data = b'{"original":"reference"}'
    digest = hashlib.sha256(data).hexdigest()
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'reference').write_bytes(data)
    original = dict(path='/Core/reference',sha256=digest)
    manifest = cache/'inputs.json'
    manifest.write_text(json.dumps(dict(members={
        original['path']: dict(file='reference',sha256=digest)})))
    spec = dict(path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert locate_input(None, original) is original
    assert read_pin(locate_input(spec, original)) == {'original':'reference'}
    with pytest.raises(ValueError, match='PORTABLE_INPUT_DOES_NOT_MATCH'):
        locate_input(spec, dict(original,sha256='0'*64))
    (cache/'reference').write_bytes(data+b' ')
    with pytest.raises(ValueError, match='RUNTIME_INPUT_PIN_CHANGED'):
        read_pin(locate_input(spec, original))


def test_pbs_deadline_uses_the_actual_grant_when_it_is_earlier():
    job = dict(stime='Wed Nov 18 20:41:19 2026',Resource_List=dict(walltime='06:00:00'))
    window = allocation_window(job,campaign_deadline=1795027583.855488)
    assert window['deadline_epoch'] == window['PBS_start_epoch'] + 21600 - 120
    assert window['deadline_epoch'] + 60 < window['allocation_deadline_epoch']
    next_start = dict(job,stime='Wed Nov 18 21:15:19 2026')
    assert allocation_window(next_start,campaign_deadline=1795027583.855488)['deadline_epoch'] == 1795027583.855488
