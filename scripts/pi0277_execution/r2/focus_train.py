"""Bind the qualified learner to the one authorized FOCUS donor and RNG namespace."""
import argparse,hashlib,json
from pathlib import Path
import train

def main(root,phase,route,limit,resume):
    card=json.loads((root/'FOCUS_AUTHORITY.json').read_text())
    assert card['status']=='ACTUAL_PRIMARY4096_DEVELOPMENT_FAIL' and card['launch_allowed']
    donor=json.loads((root/'DONOR_LOCAL_READBACK.json').read_text())
    assert donor['point']=='GRIP4096' and not donor['optimizer_inherited']
    train.DONOR_REV=donor['revision']
    namespace=json.loads((root/'design/TRAIN_PLAN.json').read_text())['noise_namespace']
    assert namespace=='PI0277:FOCUS:R2:27703:{unit}'
    train.seed_for=lambda unit:int.from_bytes(hashlib.sha256(namespace.format(unit=unit).encode()).digest()[:8],'big')%2**31
    train.run(train.Binding(root=root,arm='FOCUS',phase=phase,route=route,limit=limit,resume=resume))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--route',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--resume',type=int,default=0)
    a=p.parse_args();main(a.root,a.phase,a.route,a.limit,a.resume)
