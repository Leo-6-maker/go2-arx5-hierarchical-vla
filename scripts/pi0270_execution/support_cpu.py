"""Finite CPU label/codec work on completed TRAIN families, beside collection."""
import argparse
import fcntl
import subprocess
from admission import BASE, ROOT, read, sha, write
from collect import native_environment


def main(family):
    raw = __import__('pathlib').Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')
    with (ROOT / 'evidence/support_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        card = read(ROOT / 'design/COLLECTION_CARD.json')
        cases = [r for r in card['cases'] if r['family'] == family]
        assert len(cases) == 8
        for case in cases:
            folder = raw / 'collection/PREFIX4096' / case['case']; result = read(folder / 'result.json')
            assert result['validity'] == 'VALID' and result['collection_card_sha256'] == sha(ROOT / 'design/COLLECTION_CARD.json')
            assert sha(folder / 'trajectory.npz') == result['arrays_sha256']
        for row in read(ROOT / 'design/LABEL_FILESET.json')['files']:
            assert sha(row['path']) == row['sha256'], row['path']
        native, env = native_environment(); env['CUDA_VISIBLE_DEVICES'] = ''; env['PYTHONUNBUFFERED'] = '1'
        with (ROOT / 'evidence' / ('support-cpu-' + family + '.log')).open('ab') as log:
            subprocess.run(['/usr/bin/nice','-n','19',str(native),'-u',str(ROOT / 'code/labels.py'),'--lane','original','--family',family], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            env.pop('LD_PRELOAD',None); env.pop('LD_LIBRARY_PATH',None)
            modern=BASE/'PI0250/runtime/venv/bin/python'
            for lane in ('component','original'):
                command=['/usr/bin/nice','-n','19',str(modern),'-u',str(ROOT/'code/codec.py'),'--lane',lane]
                if lane == 'original':command+=['--family',family]
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(ROOT/'evidence'/('SUPPORT_CPU_'+family+'.json'),dict(status='PASS_COMPLETED_FAMILY_AND_COMPONENT_LABEL_CODEC_PROCESSES',
              family=family,component_proposals=640,original_family_proposals=256,model_queries=0,render_calls=0,physics_steps=0,
              source_sha256=sha(__file__),all24_data_admission=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);main(p.parse_args().family)
