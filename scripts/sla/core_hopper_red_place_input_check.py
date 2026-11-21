"""Verify every input in the original mixed TRAIN-020 using the extended producer registry."""
import argparse
import json
import time
from pathlib import Path

from core_hopper_mixed_readback import inputs
from hopper_reader_compat import RED_PLACE_MANIFEST, reader_identity, sha
from scripts.sla.hopper_parent_custody import PLAN, write_once
from scripts.sla.hopper_site import pinned


def check(cfg):
    if (cfg['plan']!=PLAN or cfg['entry']['sha256']!='ea58638ac41327fc25400993d9e593d46b881c49764228cddbe8ad87dc2b85d9'
            or reader_identity(Path(cfg['capsule'])/'candidate')[0]!=RED_PLACE_MANIFEST):
        raise ValueError('Unexpected original twenty examples or candidate reader')
    old=pinned(cfg['previous_readback'])['actual']
    if old['episodes']!=20 or old['windows']!=1767 or old['seal']!=cfg['entry']['sha256']:
        raise ValueError('Incorrect baseline readback')
    start=time.process_time()
    record=inputs(dict(cfg,mode='candidate'),cfg['entry'],old)
    if record!=old:
        raise ValueError('The extended producer registry changed a previously accepted input')
    return dict(plan=PLAN,status='TRAIN020_EXTENDED_READER_EQUAL',reader_manifest=RED_PLACE_MANIFEST,
                previous_readback_sha256=cfg['previous_readback']['sha256'],actual=record,
                cpu_seconds=time.process_time()-start,newly_admitted_examples=0,
                native_requests=0,optimizer_updates=0,model_generations=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('config',type=Path)
    cli.add_argument('config_sha256')
    args=cli.parse_args()
    cfg=pinned(dict(path=str(args.config),sha256=args.config_sha256))
    if sha(Path(__file__))!=cfg['worker']:
        raise ValueError('Mounted input qualification source differs')
    dest=args.config.parent/'TRAIN020_READBACK.json'
    write_once(dest,check(cfg))
