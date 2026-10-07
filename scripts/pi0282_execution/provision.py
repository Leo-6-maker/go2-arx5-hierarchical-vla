"""Local-only TRAIN reuse and per-file quota evidence, no model or GPU calls."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main(root):
    root=Path(root);data=root/'data';data.mkdir(parents=True,exist_ok=True)
    contract=json.loads((data/'DATA_CONTRACT.json').read_text())
    expected=json.loads((data/'RAW_MATERIALIZATION.json').read_text())
    rows={e['episode']:e for e in expected['episodes']};receipts=[]
    old=root.parent/'PI0276/source/data'
    started=time.monotonic()
    for episode in contract['admitted_episodes']:
        if episode['split']!='TRAIN':continue
        name='TRAIN-'+str(episode['seed']);parent=old/name;target=data/'episodes'/name;target.mkdir(parents=True,exist_ok=True)
        assert sha(parent/'commands.npz')==episode['command_sha256'] and sha(parent/'observations.npz')==episode['observation_sha256']
        verified={}
        for archive,keys in [('commands.npz',('action',)),('observations.npz',('state_v1','base_camera_raw','hand_camera_raw'))]:
            with np.load(parent/archive) as source:
                for key in keys:
                    output=target/(key+'.npy');wanted=rows[name]['files'][key]
                    if not output.exists():
                        value=source[key]
                        assert hashlib.sha256(value.tobytes()).hexdigest()==wanted['array_sha256']
                        partial=output.with_suffix('.partial')
                        with partial.open('wb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
                        assert partial.stat().st_size==wanted['bytes'] and sha(partial)==wanted['sha256']
                        partial.replace(output)
                    assert output.stat().st_size==wanted['bytes'] and sha(output)==wanted['sha256']
                    verified[key]=dict(path=str(output),sha256=wanted['sha256'],bytes=wanted['bytes'])
        receipts.append(dict(episode=name,original=str(parent),files=verified));print('REUSED_RAW',name,flush=True)
    assert len(receipts)==93
    report=dict(status='PASS_ORIGINAL_93_TRAIN_LOCAL_RAW_MEMBER_READBACK',elapsed_s=time.monotonic()-started,episodes=receipts,
        original_source_archives_untouched=True,total_new_bytes=sum(f['bytes'] for e in receipts for f in e['files'].values()),new_model_downloads=0)
    (data/'HOPPER_RAW_READBACK.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();main(args.root)
