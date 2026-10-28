"""New fixed final S4 model-only views; no optimizer copies."""
import argparse
import os
import sys
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0273'
online=ROOT/'runtime/hf_online'
os.environ.update(HF_HOME=str(online),HF_XET_CACHE=str(online/'xet'),HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=[str(BASE/'PI0264/code'),str(BASE/'PI0263/code')]
from pi0264_plan import read,sha,write


def main(point):
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');assert audit['status']=='PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING'
    donor=read(BASE/'PI0270/model_bindings/AGG4096.json')
    assert donor['source_revision']=='leoyuer/pi0246-gate-pass-campaign@f862abb26041c8e7d355470d6e9d48899f78b9ce/arms/PI0270/AGG/checkpoint-4096'
    assert donor['checkpoint_tree_sha256']=='0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66'
    from huggingface_hub import hf_hub_download
    repo,revision_prefix=donor['source_revision'].split('@',1);revision,prefix=revision_prefix.split('/',1)
    donor_root=Path(donor['view']).parents[3]
    for name,digest in donor['model_files_sha256'].items():
        path=Path(donor['view'])/name
        if not path.exists():assert Path(hf_hub_download(repo,filename=prefix+'/'+name,revision=revision,local_dir=donor_root))==path
        assert sha(path)==digest
    arm=point.removesuffix('5120');assert arm in ('BROAD','TARGET')
    mode=ROOT/'design'/f'{arm}_READOUT_R1.json'
    source=read(ROOT/'evidence'/arm/'checkpoint-5120.archive_receipt.json')
    assert source['status']=='UPLOADED' and (source['arm'],source['step'])==(arm,5120)
    entries={r['path']:r for r in source['entries']}
    checkpoint=next(c for c in audit['complete_checkpoints'] if (c['arm'],c['step'])==(arm,5120))
    assert entries=={r['path']:r for r in checkpoint['entries']}
    names=('config.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json')
    download=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273/models')/point;view=download/source['prefix']
    for name in names:
        p=view/name
        if not p.exists():assert Path(hf_hub_download(source['repo_id'],filename=source['prefix']+'/'+name,revision=source['revision'],local_dir=download))==p
        assert p.stat().st_size==entries[name]['bytes'] and sha(p)==entries[name]['sha256']
    binding=dict(schema='PI0273_FIXED_MODEL_ONLY_R1',point=point,view_type='model-only',view=str(view),
        source_revision=f"{source['repo_id']}@{source['revision']}/{source['prefix']}",checkpoint_tree_sha256=source['tree_sha256'],
        model_files_sha256={name:entries[name]['sha256'] for name in names},effective_contract='ROT',
        statistics_sha256=sha(ROOT/'contracts/ROT/source/dataset/statistics.json'),readout_contract_sha256=sha(mode))
    import pi0264_checkpoint_scope as scope
    before=scope.tensors(Path(donor['view']),donor);after=scope.tensors(view,binding)
    assert set(before)==set(after) and len(after)==1030
    cfg=read(view/'config.json');assert all(cfg[k] is False for k in ('tune_llm','tune_visual','tune_vlln'))
    assert all(cfg[k] is True for k in ('tune_projector','tune_diffusion_model')) and cfg['num_inference_timesteps']==4 and cfg['action_horizon']==40
    frozen=[];trainable=[]
    for name,value in after.items():
        assert all(value[k]==before[name][k] for k in ('dtype','shape','bytes'))
        if name.startswith(scope.FROZEN):assert value==before[name],name;frozen.append(name)
        else:assert name.startswith('action_head.');trainable.append(dict(name=name,changed=value['sha256']!=before[name]['sha256']))
    assert len(frozen)==559 and len(trainable)==471
    write(ROOT/'evidence'/point/'FROZEN_SCOPE.json',dict(status='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE',
        binding=binding,donor=donor,frozen=frozen,trainable=trainable,model_queries=0,gradient_queries=0))
    assert binding['statistics_sha256']==read(mode)['statistics_sha256']
    write(ROOT/'model_bindings'/f'{point}.json',binding)
    print('PASS_PI0273_FIXED_MODE_LABELLED_MODEL_ONLY_VIEW',point,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','TARGET5120'),required=True);main(p.parse_args().point)
