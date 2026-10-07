"""One new COVER93 optimizer lineage, with the parent's qualified final objective."""
import argparse,hashlib,importlib.util,sys
from pathlib import Path
import cpu_inputs as data
import train

def main(root,phase,limit,resume):
    c=data.read(root/'RUN_CONFIG.json');assert c['release']==train.RELEASE and c['launch_allowed']
    assert __import__('os').environ.get('PBS_JOBID'),'Donor readback/model work belongs inside the allocation'
    donor=data.read(root/'DONOR_LOCAL_READBACK.json');marker=root/'evidence/DONOR_CURRENT_MODEL_READBACK.json'
    if not marker.exists():
        files=[]
        for r in donor['files']:
            p=Path(donor['path'])/r['path'];assert p.stat().st_size==r['bytes'] and data.sha(p)==r['sha256']
            s=p.stat();files.append(dict(r,inode=s.st_ino,dev=s.st_dev,mtime_ns=s.st_mtime_ns))
        data.write(marker,dict(status='PASS_ACTUAL_FOCUS4096_MODEL_FILE_READBACK',revision=donor['revision'],files=files,optimizer_inherited=False))
    for r in data.read(marker)['files']:
        s=(Path(donor['path'])/r['path']).stat();assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(r['dev'],r['inode'],r['mtime_ns'],r['bytes'])
    namespace=data.read(root/'design/TRAIN_PLAN.json')['noise_namespace'];assert namespace=='PI0278:COVER93:27803:{unit}'
    train.seed_for=lambda unit:int.from_bytes(hashlib.sha256(namespace.format(unit=unit).encode()).digest()[:8],'big')%2**31
    spec=importlib.util.spec_from_file_location('qualified_bound',c['bound_source']);bound=importlib.util.module_from_spec(spec);spec.loader.exec_module(bound)
    bound.PHASE='PI0278_COVER93_FINAL_PARENT_OBJECTIVE';sys.path.insert(0,str(Path(c['bound_source']).parent));train.add_box=bound.add_box
    train.run(train.Binding(root,'COVER93',phase,'FAST_CACHE2',limit,resume))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--resume',type=int,default=0);a=p.parse_args();main(a.root,a.phase,a.limit,a.resume)
