"""Finite CPU closeout of the 24 frozen collection families; no extra sampling."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from admission import BASE, ROOT, read, sha, write
from support_cpu import main as support

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def collector_live(pid):
    path = Path('/proc')/str(pid)/'cmdline'
    try:
        return str(ROOT/'code/collect.py').encode() in path.read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def main(check_only=False):
    card = read(ROOT/'design/COLLECTION_CARD.json')
    families = list(dict.fromkeys(r['family'] for r in card['cases']))
    assert len(families) == 24 and len(card['cases']) == 192
    assert all(sum(r['family'] == f for r in card['cases']) == 8 for f in families)
    pid = read(ROOT/'evidence/COLLECTION_PROCESS_START.json')['pid']
    sources = [ROOT/'code'/n for n in ('finish_support.py', 'support_cpu.py', 'freeze_training_data.py')]
    frozen = {str(p): sha(p) for p in sources}
    if check_only:
        assert collector_live(pid) or (ROOT/'evidence/COLLECTION_PROCESS_COMPLETE.json').exists()
        print('PASS_FIXED_24_FAMILY_FINITE_CPU_ROUTE_NO_CALLS', flush=True); return
    with (ROOT/'evidence/support_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for family in families:
            receipt = ROOT/'evidence'/('SUPPORT_CPU_'+family+'.json')
            if receipt.exists():
                assert read(receipt)['status'] == 'PASS_COMPLETED_FAMILY_AND_COMPONENT_LABEL_CODEC_PROCESSES'
                continue
            cases = [r for r in card['cases'] if r['family'] == family]
            while not all((RAW/'collection/PREFIX4096'/c['case']/'result.json').exists() for c in cases):
                assert collector_live(pid), ('COLLECTOR_TERMINAL_WITH_MISSING_FIXED_FAMILY', pid, family)
                time.sleep(30)
            assert all(sha(p) == frozen[str(p)] for p in sources)
            support(family)
        while not (ROOT/'evidence/COLLECTION_PROCESS_COMPLETE.json').exists():
            assert collector_live(pid), ('COLLECTOR_COMPLETION_RECEIPT_MISSING', pid)
            time.sleep(30)
        proposals = read(ROOT/'design/LABEL_PROPOSAL_CARD.json')['proposals']
        assert len(proposals) == 6784
        for row in proposals:
            path = RAW/'codec'/row['source_lane']/row['source_episode_key']/f"{row['query_index']:03d}.json"
            assert path.is_file(), row['uid']
        write(ROOT/'evidence/ALL_FIXED_SUPPORT_CPU_COMPLETE.json', dict(
            status='PASS_ALL24_FAMILIES_AND_ALL6784_CODEC_PROPOSALS', families=families,
            proposals=6784, sources=frozen, new_model_queries=0, new_render_calls=0,
            new_optimizer_updates=0, data_admission_not_assumed=True))
        assert all(sha(p) == frozen[str(p)] for p in sources)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        subprocess.run(['/usr/bin/nice', '-n', '19', str(BASE/'PI0250/runtime/venv/bin/python'),
                        '-u', str(ROOT/'code/freeze_training_data.py')], env=env, check=True)
        print('FIXED_SUPPORT_AND_DATA_ADMISSION_PROCESS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true')
    main(parser.parse_args().check_only)
