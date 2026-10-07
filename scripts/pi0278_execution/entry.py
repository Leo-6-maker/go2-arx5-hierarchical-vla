"""One new COVER93 optimizer lineage, with the parent's qualified final objective."""
import argparse,hashlib,importlib.util,sys
from pathlib import Path
import cpu_inputs as data
import train

def main(root,phase,limit,resume):
    c=data.read(root/'RUN_CONFIG.json');assert c['release']==train.RELEASE and c['launch_allowed']
    namespace=data.read(root/'design/TRAIN_PLAN.json')['noise_namespace'];assert namespace=='PI0278:COVER93:27803:{unit}'
    train.seed_for=lambda unit:int.from_bytes(hashlib.sha256(namespace.format(unit=unit).encode()).digest()[:8],'big')%2**31
    spec=importlib.util.spec_from_file_location('qualified_bound',c['bound_source']);bound=importlib.util.module_from_spec(spec);spec.loader.exec_module(bound)
    bound.PHASE='PI0278_COVER93_FINAL_PARENT_OBJECTIVE';sys.path.insert(0,str(Path(c['bound_source']).parent));train.add_box=bound.add_box
    train.run(train.Binding(root,'COVER93',phase,'FAST_CACHE2',limit,resume))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--resume',type=int,default=0);a=p.parse_args();main(a.root,a.phase,a.limit,a.resume)
