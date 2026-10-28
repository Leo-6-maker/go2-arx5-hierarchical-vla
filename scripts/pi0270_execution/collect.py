"""One finite serialized collection owner; never a monitor or second worker."""
import fcntl
import os
import subprocess
import sys
from admission import BASE, ROOT, read, sha, write


def native_environment():
    sys.path.insert(0, str(BASE / 'PI0256/runtime'))
    import pi0255_run
    pi0255_run.ROOT = BASE / 'PI0267'
    py, env = pi0255_run.environment()
    # Keep the qualified native IsaacGym library/module paths when prepending this entry.
    env['PYTHONPATH'] = ':'.join([str(ROOT / 'code'), str(BASE / 'PI0267/code'), str(BASE / 'PI0266/code'), env['PYTHONPATH']])
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    return py, env


def main():
    with (ROOT / 'evidence/collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(ROOT / 'evidence/A0_PREDECESSOR_CLOSEOUT.json')['status'] == 'PASS_CONTINUE_PI0270_A1'
        seal = read(ROOT / 'design/COLLECTION_FILESET.json')
        for row in seal['files']:
            assert sha(row['path']) == row['sha256'], row['path']
        card = read(ROOT / 'design/COLLECTION_CARD.json')
        assert len(card['cases']) == 192 and card['collector'] == 'PREFIX4096'
        py, env = native_environment()
        families = list(dict.fromkeys(r['family'] for r in card['cases']))
        assert len(families) == 24
        for family in families:
            with (ROOT / 'evidence' / ('collection-' + family + '.log')).open('ab') as log:
                subprocess.run([str(py), str(ROOT / 'code/collect_episode.py'), '--point', 'PREFIX4096', '--family', family], env=env, stdout=log, stderr=log, check=True)
        write(ROOT / 'evidence/COLLECTION_PROCESS_COMPLETE.json', dict(status='ALL_FIXED_COLLECTION_FAMILY_PROCESSES_EXIT_ZERO',
              cases=192, families=families, collection_card_sha256=sha(ROOT / 'design/COLLECTION_CARD.json'),
              selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
        print('ALL_FIXED_COLLECTION_FAMILY_PROCESSES_EXIT_ZERO', flush=True)


if __name__ == '__main__':
    main()
