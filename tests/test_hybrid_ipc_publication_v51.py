import json

import pytest

from scripts.sla import hybrid_ipc_worker_v3 as worker


def test_transient_missing_publication_does_not_create_or_replay_a_request(monkeypatch):
    message={'request':{'request_id':'original'},'response_root':'same-response'}
    calls=[]
    def read(path):
        calls.append(path)
        if len(calls)==1:raise FileNotFoundError('publication temporarily unavailable')
        return message
    monkeypatch.setattr(worker,'read',read)
    assert worker.poll_message('REQUEST.json')is None
    assert worker.poll_message('REQUEST.json')is message
    assert calls==['REQUEST.json','REQUEST.json']


@pytest.mark.parametrize('error',[json.JSONDecodeError('bad JSON','{',0),PermissionError('not readable'),OSError('I/O failure')])
def test_publication_only_retries_absence(monkeypatch,error):
    def read(path):raise error
    monkeypatch.setattr(worker,'read',read)
    with pytest.raises(type(error)):
        worker.poll_message('REQUEST.json')
