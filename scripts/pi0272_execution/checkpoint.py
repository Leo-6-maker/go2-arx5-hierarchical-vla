"""Only new formal PI0272 Trainers, using the qualified full-byte HF helper."""
import argparse
import json
import os
import sys
from pathlib import Path


def main(arm,step):
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0272')
    cp=root/arm/'main/training'/f'checkpoint-{step}'
    mode=json.loads((cp/'PI0272_READOUT_CONTRACT.json').read_text())
    assert mode['mode']==arm and mode['generated_nodes']==(1 if arm=='DIRECT' else 4)
    assert (cp/'pi0272_provenance/EXECUTION_FILESET.json').is_file()
    online=root/'runtime/hf_online'
    os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',
                      TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
    sys.path.insert(0,str(root.with_name('PI0267')/'code'))
    import pi0264_archive_trainer as mature
    mature.ROOT=root
    mature.main(arm,step,max_steps=1024,campaign='PI0272',token=(Path.home()/'.config/huggingface/token').read_text().strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('S4FIT','DIRECT'),required=True)
    p.add_argument('--step',type=int,choices=(512,1024),required=True);a=p.parse_args();main(a.arm,a.step)
