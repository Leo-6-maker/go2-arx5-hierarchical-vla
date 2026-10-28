"""Archive only AGG2048/4096 complete Trainers with the qualified helper."""
import argparse
import os
from pathlib import Path


def main(step):
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0270');online=root/'runtime/hf_online'
    os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
    import sys
    sys.path.insert(0,str(root.with_name('PI0267')/'code'))
    import pi0264_archive_trainer as mature
    mature.ROOT=root
    mature.main('AGG',step,max_steps=4096,campaign='PI0270',token=(Path.home()/'.config/huggingface/token').read_text().strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(2048,4096),required=True);main(p.parse_args().step)
