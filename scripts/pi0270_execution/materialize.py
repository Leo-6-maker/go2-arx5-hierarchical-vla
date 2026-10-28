"""Only final model views on Corelab; complete Trainer states stay Hopper/HF."""
import argparse
import os
import sys
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0270'
online=ROOT/'runtime/hf_online'
os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=[str(BASE/'PI0264/code'),str(BASE/'PI0263/code')]
from pi0264_plan import read,sha,write
from huggingface_hub import hf_hub_download

def main(arm):
    assert arm=='AGG'
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');assert audit['status']=='PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY'
    source=read(ROOT/'evidence'/arm/'checkpoint-4096.full_readback.json')
    assert source['status']=='PASS_FULL_FILE_READBACK' and source['arm']==arm and source['step']==4096
    entries={r['path']:r for r in source['files']}
    checkpoint=next(c for c in audit['complete_checkpoints'] if c['arm']==arm and c['step']==4096)
    assert entries=={r['path']:r for r in checkpoint['entries']} and len(entries)==18
    names=('config.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json')
    download=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270/models')/(arm+'4096');view=download/source['prefix']
    for name in names:
        p=view/name
        if not p.exists():assert Path(hf_hub_download(source['repo_id'],filename=source['prefix']+'/'+name,revision=source['revision'],local_dir=download))==p
        assert p.stat().st_size==entries[name]['bytes'] and sha(p)==entries[name]['sha256']
    binding=dict(schema='PI0270_FIXED_FINAL_MODEL_ONLY_V1',point=arm+'4096',view_type='model-only',view=str(view),
        source_revision=f"{source['repo_id']}@{source['revision']}/{source['prefix']}",checkpoint_tree_sha256=source['tree_sha256'],
        model_files_sha256={name:entries[name]['sha256'] for name in names},effective_contract='ROT',
        statistics_sha256=sha(ROOT/'contracts/ROT/source/dataset/statistics.json'))
    write(ROOT/'model_bindings'/(arm+'4096.json'),binding)
    import pi0264_checkpoint_scope as scope
    donor=read(BASE/'PI0268/R1/model_bindings/ROT2048.json')
    before=scope.tensors(Path(donor['view']),donor);after=scope.tensors(view,binding)
    assert set(before)==set(after) and len(after)==1030
    cfg=read(view/'config.json');assert all(cfg[k] is False for k in ('tune_llm','tune_visual','tune_vlln'))
    assert all(cfg[k] is True for k in ('tune_projector','tune_diffusion_model')) and cfg['num_inference_timesteps']==4 and cfg['action_horizon']==40
    frozen=[];trainable=[]
    for name,value in after.items():
        assert value['dtype']==('BF16' if name.startswith('backbone.') else 'F32')
        assert all(value[k]==before[name][k] for k in ('dtype','shape','bytes'))
        if name.startswith(scope.FROZEN):assert value==before[name],name;frozen.append(name)
        else:assert name.startswith('action_head.');trainable.append(dict(name=name,changed=value['sha256']!=before[name]['sha256']))
    assert len(frozen)==559 and len(trainable)==471
    write(ROOT/'evidence'/(arm+'4096')/'FROZEN_SCOPE.json',dict(status='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE',binding=binding,
         donor=donor,frozen=frozen,trainable=trainable,model_queries=0,gradient_queries=0))
    print('PASS_FIXED_FINAL_MODEL_VIEW',arm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('AGG',));main(p.parse_args().arm)
