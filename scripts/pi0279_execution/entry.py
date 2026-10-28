"""One CLEAR segment, reusing the qualified training/loss implementation."""
import argparse,hashlib,importlib.util,os,sys
from pathlib import Path
import cpu_inputs as data
import train
import persistence  # Bind this run's creator before inherited module paths are prepended.
RELEASE='4afdb1401650353092428dcb69d0cb353ee74164'


def segment(root,phase,limit,resume):
    assert os.environ.get('PBS_JOBID')
    settings=data.read(root/'RUN_CONFIG.json');assert settings['release']==RELEASE and settings['launch_allowed']
    donor=data.read(root/'DONOR_LOCAL_READBACK.json');marker=root/'evidence/DONOR_CURRENT_MODEL_READBACK.json'
    if not marker.exists():
        files=[]
        for row in donor['files']:
            p=Path(donor['path'])/row['path'];assert p.stat().st_size==row['bytes'] and data.sha(p)==row['sha256']
            s=p.stat();files.append(dict(row,inode=s.st_ino,dev=s.st_dev,mtime_ns=s.st_mtime_ns))
        data.write(marker,dict(status='PASS_ACTUAL_COVER934096_MODEL_READBACK',files=files,revision=donor['revision'],optimizer_inherited=False))
    for row in data.read(marker)['files']:
        s=(Path(donor['path'])/row['path']).stat();assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(row['dev'],row['inode'],row['mtime_ns'],row['bytes'])
    namespace=data.read(root/'design/TRAIN_PLAN.json')['noise_namespace'];assert namespace=='PI0279:CLEAR:27903:{unit}'
    train.RELEASE=RELEASE;train.seed_for=lambda unit:int.from_bytes(hashlib.sha256(namespace.format(unit=unit).encode()).digest()[:8],'big')%2**31
    spec=importlib.util.spec_from_file_location('pi0279_qualified_box',settings['bound_source']);bound=importlib.util.module_from_spec(spec);spec.loader.exec_module(bound)
    bound.PHASE='PI0279_CLEAR_INHERITED_FULL_PARENT_OBJECTIVE';sys.path.insert(0,str(Path(settings['bound_source']).parent));train.add_box=bound.add_box
    train.run(train.Binding(root,'CLEAR',phase,'FAST_CACHE2',limit,resume))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--resume',type=int,default=0);a=p.parse_args();segment(a.root,a.phase,a.limit,a.resume)
