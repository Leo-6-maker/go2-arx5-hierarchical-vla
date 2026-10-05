"""Formal2560/5120 full Trainers through the original pinned HF archive helper."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0274')
def main(step):
    cp=ROOT/'EQUIL/main/training'/f'checkpoint-{step}';mode=json.loads((cp/'PI0274_READOUT_CONTRACT.json').read_text())
    assert mode['mode']=='S4' and mode['arm']=='EQUIL' and mode['generated_nodes']==4
    assert (cp/'pi0274_provenance/EXECUTION_FILESET_R1.json').is_file()
    online=ROOT/'runtime/hf_online';os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
    sys.path.insert(0,str(ROOT.with_name('PI0267')/'code'));import pi0264_archive_trainer as mature
    mature.ROOT=ROOT;mature.main('EQUIL',step,max_steps=5120,campaign='PI0274',token=(Path.home()/'.config/huggingface/token').read_text().strip())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(2560,5120),required=True);main(p.parse_args().step)
