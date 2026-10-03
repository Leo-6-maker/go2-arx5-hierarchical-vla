"""Reuse full-Trainer archiving and serialized scope checks; never load optimizer on Corelab."""
import argparse
import json
import os
from pathlib import Path


def archive(arm,step):
    online=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0269/runtime/hf_online')
    os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
    import sys
    sys.path.insert(0,'/scratch/e1192548/Projects/go2-arx5/experiments/PI0267/code')
    import pi0264_archive_trainer as old
    old.ROOT=old.ROOT.with_name('PI0269')
    old.main(arm,step,max_steps=4096,campaign='PI0269',token=(Path.home()/'.config/huggingface/token').read_text().strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('UNIFORM','PREFIX'),required=True)
    p.add_argument('--step',type=int,choices=(2048,4096),required=True);a=p.parse_args();archive(a.arm,a.step)
