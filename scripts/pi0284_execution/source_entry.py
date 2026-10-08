"""Versioned entry/accounting adapter; original physics, reference installation and score."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
from budget import publication,process

def run(root,point,panel):
    root=Path(root);publication(root);card=json.loads((root/'design/RUNTIME_CARD.json').read_text());sys.path.append(card['parent_code'])
    from prepare_core import sha
    for path,h in card['dependencies'].items():assert sha(path)==h,path
    spec=importlib.util.spec_from_file_location('pi0283_fixed_source',Path(card['parent_code'])/'source.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    source.ROOT=root;source.OUT=root/'source'
    original_setup=source.setup
    def setup():
        worker=original_setup()
        worker.LIMITS=dict(starts=64,control=45000,physics=230000,rgb=45000,teacher=0,dispatch=90000)
        return worker
    source.setup=setup
    original_step=source.step
    def step(worker,ep,command):
        ep.ledger.reserve('dispatch',ep.slot);ep.ledger.reserve('dispatch',ep.slot)
        return original_step(worker,ep,command)
    source.step=step
    started=process(root,'start')
    try:source.run(point,panel)
    finally:process(root,'end',started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',required=True);p.add_argument('--panel',choices=('TRAIN','DEV','CONFIRM'),required=True);a=p.parse_args();run(a.root,a.point,a.panel)
