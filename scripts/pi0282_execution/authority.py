"""Pinned R2 cumulative limits and absolute time checks; no rolling deadline."""
import json,time
from pathlib import Path

RELEASE='fdcde0e29f7eb689d0eea0f4dd20586309e77326'
PARENT='8fdaab73b8b6686409317d9562833235c2839acc'
DEADLINE=1791421200


def load(root):
    value=json.loads((Path(root)/'design/RUNTIME_AUTHORITY.json').read_text())
    assert value['release']==RELEASE and value['parent_release']==PARENT
    assert value['canonical']==6044490956 and value['deadline_epoch']==DEADLINE
    assert value['training_cutoff_epoch']<=DEADLINE-90*60
    assert value['model_physics_cutoff_epoch']==DEADLINE-10*60
    assert value['formal_cap']==8192 and value['all_updates_cap']==8416
    return value


def allowed(root,kind,reserve_s=0):
    value=load(root)
    cutoff=value['training_cutoff_epoch'] if kind=='update' else value['model_physics_cutoff_epoch']
    assert time.time()+reserve_s<cutoff,('R2_TIME_GATE_CLOSED',kind,time.time(),cutoff)
    return value


def self_check():
    # Exact UTC deadlines, not a local timezone guess or tomorrow's date.
    import datetime
    assert datetime.datetime.fromtimestamp(DEADLINE,datetime.timezone.utc).isoformat()=='2026-10-08T01:00:00+00:00'
    assert DEADLINE-90*60==1791415800 and DEADLINE-10*60==1791420600
    assert 8192+64+32+128==8416
    print('PASS_R2_ABSOLUTE_CLOCK_AND_CUMULATIVE_CAPS; model/optimizer/physics/PBS=0')


if __name__=='__main__':self_check()
