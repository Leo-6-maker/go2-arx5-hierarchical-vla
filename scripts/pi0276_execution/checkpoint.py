"""Full task Trainer upload with the qualified fixed-revision byte readback."""
import argparse,json,os,sys
from pathlib import Path

def main(step):
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0276');cp=root/'TASK/main/training'/f'checkpoint-{step}'
    mode=json.loads((cp/'PI0276_READOUT_CONTRACT.json').read_text())
    assert mode['mode']=='S4' and mode['arm']=='TASK' and mode['generated_nodes']==4
    assert (cp/'pi0276_provenance/EXECUTION_FILESET.json').is_file()
    online=root/'runtime/hf_online'
    os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_PROGRESS_BARS='1')
    sys.path.insert(0,str(root.with_name('PI0267')/'code'))
    import pi0264_archive_trainer as mature
    mature.ROOT=root;mature.main('TASK',step,max_steps=8192,campaign='PI0276',token=(Path.home()/'.config/huggingface/token').read_text().strip())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(4096,8192),required=True);main(p.parse_args().step)
