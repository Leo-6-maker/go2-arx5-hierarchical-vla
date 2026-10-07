"""Configure isolated evaluation roots before importing the qualified components."""
import sys
from pathlib import Path
COMMON=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278');ROOT=COMMON/'R2'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2')
OLD=COMMON.with_name('PI0276');OLD_RAW=RAW.parent.with_name('PI0276')
RELEASE='83af9099f3bab9a4ac08eaa85380c2cfd6351f34'

def bind():
    sys.path.insert(0,str(ROOT/'code'))
    import evaluation_card as c
    c.COMMON,c.ROOT,c.RAW,c.OLD,c.OLD_RAW,c.RELEASE=COMMON,ROOT,RAW,OLD,OLD_RAW,RELEASE
    return c
