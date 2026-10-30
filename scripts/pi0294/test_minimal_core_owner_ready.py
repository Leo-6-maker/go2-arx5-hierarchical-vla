"""Real local process readiness; no model, GPU or scientific request."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.pi0294.generalization.minimal_core_owner import await_fixture_sessions


@pytest.fixture
def local_server(tmp_path):
    root=tmp_path/'registered-child';root.mkdir()
    label='PV1-PREMODEL-ALLOCATION-01'
    path=root/'aux_worker_processes'/('fixture-rpc-hopper-'+label+'.json')
    program='''import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]);p.parent.mkdir();time.sleep(.25)
v=dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha='sealed-source',job_id='own-job',source_live_worker=False,work_type='PV1_ISOLATED_TRANSPORT_NO_MODEL_NO_PHYSICS',maximum_CPU_s=180)
p.write_text(json.dumps(v));sys.stdin.read()
'''
    process=subprocess.Popen([sys.executable,'-B','-c',program,str(path)],stdin=subprocess.PIPE)
    session=SimpleNamespace(role='hopper',label=label,process=process)
    try:yield session,dict(hopper=dict(root=str(root),source_sha='sealed-source')),path
    finally:
        process.stdin.close()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate();process.wait(timeout=3)


def test_waits_for_initialized_exact_process_before_first_request(local_server):
    session,sites,path=local_server;beats=[]
    await_fixture_sessions([session],sites,lambda:beats.append(time.monotonic()),
        'own-job',deadline=time.time()+3)
    assert path.exists() and len(beats)>1 and session.process.poll()is None


def test_expired_initialization_is_terminal_without_sending_request(local_server):
    session,sites,path=local_server
    with pytest.raises(TimeoutError,match='NO_REQUEST_SENT'):
        await_fixture_sessions([session],sites,lambda:None,'own-job',deadline=time.time()-1)
    assert not path.exists()


def test_ready_record_does_not_replace_exact_source_identity(local_server):
    session,sites,path=local_server;sites['hopper']['source_sha']='wrong-source'
    with pytest.raises(PermissionError,match='EXACT_EXISTING_LOCAL_RPC_READY_PROCESS'):
        await_fixture_sessions([session],sites,lambda:None,'own-job',deadline=time.time()+3)
