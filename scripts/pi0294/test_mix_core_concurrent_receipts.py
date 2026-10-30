"""A live onceIO writer must not look like a durable orphan to a reader."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import multiprocessing

import pytest

from scripts.pi0294.test_mix_core_delegated_runtime import running, full, captured_global
from scripts.pi0294.pv1 import mix_accounting as a
from scripts.pi0294.pv1 import mix_core_delegated_control as control
from scripts.pi0294.pv1 import mix_core_delegation_store as store


@pytest.mark.parametrize('kind', ['control', 'effect'])
def test_reader_waits_for_existing_writer_and_then_checks_all_receipts(running, kind):
    """Pause a real child after fsync(intent), while its original lock is held."""
    v = running
    ctx = multiprocessing.get_context('fork')
    written, release = ctx.Event(), ctx.Event()

    def writer():
        original = control.once._exclusive
        def pause(path, raw, mode=0o400):
            original(path, raw, mode)
            if str(path).endswith('.INTENT.json'):
                written.set()
                if not release.wait(10):
                    raise TimeoutError('TEST_WRITER_RELEASE_NOT_RECEIVED')
        control.once._exclusive = pause
        if kind == 'control':
            before = control.read_confirmed(v.root, v.value['grant_bytes'])
            control.commit(v.root, v.value['grant_bytes'], kind='METRICS',
                values={'CPU_s': .5, 'RPC_calls': 1}, actor_pid=42,
                actor_start=43, now_epoch=1207.,
                expected_before_sha256=a.sha(a.raw(before)))
        else:
            p = v.process
            store.consume_once(v.root, expected_grant_sha256=a.sha(v.value['grant_bytes']),
                effect_id=f"{p['pid']}:{p['proc_start']}:0:DEBIT", kind='model_loads', count=1,
                native_facts_bytes=a.raw(p), now_epoch=1207.)

    def reader():
        if kind == 'control':
            return control.read_confirmed(v.root, v.value['grant_bytes'])
        return store.read_confirmed(v.root,
            expected_grant_sha256=a.sha(v.value['grant_bytes']))['state']

    child = ctx.Process(target=writer)
    child.start()
    try:
        assert written.wait(5), 'writer must reach the actual original intent'
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(reader)
            try:
                with pytest.raises(FutureTimeout):
                    pending.result(timeout=.2)
            finally:
                release.set()
            value = pending.result(timeout=5)
        assert len(value['events']) == 1
        if kind == 'control':
            assert value['RPC_calls'] == 1 and value['CPU_s'] == .5
        else:
            assert value['sequence'] == 1
    finally:
        release.set()
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0
