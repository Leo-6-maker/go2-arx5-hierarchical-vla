"""Reuse full-Trainer archiving and serialized scope checks; never load optimizer on Corelab."""
import argparse
import json
from pathlib import Path


def archive(arm,step):
    import pi0264_archive_trainer as old
    old.ROOT=old.ROOT.with_name('PI0267')
    old.main(arm,step,max_steps=4096,campaign='PI0267')


def materialize(arm):
    from huggingface_hub import hf_hub_download
    import pi0264_checkpoint_scope as scope
    from pi0264_plan import read,sha,write
    base=Path('/home/liuyu-nfs/Projects/go2-arx5');root=base/'PI0267/R1'
    raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
    source=read(root/'evidence'/arm/'checkpoint-4096.full_readback.json')
    assert source['status']=='PASS_FULL_FILE_READBACK' and source['complete_trainer'] and source['step']==4096
    entries={r['path']:r for r in source['files']}
    names=('config.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json')
    download=raw/'models'/(arm+'4096');view=download/source['prefix']
    for name in names:
        target=view/name
        if not target.exists():
            assert Path(hf_hub_download(source['repo_id'],filename=source['prefix']+'/'+name,revision=source['revision'],local_dir=download))==target
        assert target.stat().st_size==entries[name]['bytes'] and sha(target)==entries[name]['sha256']
    point=arm+'4096';binding=dict(schema='PI0267_FIXED_MODEL_ONLY_VIEW_V1',point=point,view_type='model-only',view=str(view),
        source_revision=f"{source['repo_id']}@{source['revision']}/{source['prefix']}",checkpoint_tree_sha256=source['tree_sha256'],
        model_files_sha256={n:entries[n]['sha256'] for n in names})
    path=root/'model_bindings'/(point+'.json')
    if path.exists():assert read(path)==binding
    else:write(path,binding)
    donor=read(root/'model_bindings/donor.json');original=scope.tensors(Path(donor['view']),donor);final=scope.tensors(view,binding)
    config=read(view/'config.json')
    assert all(config[k] is False for k in ('tune_llm','tune_visual','tune_vlln'))
    assert all(config[k] is True for k in ('tune_projector','tune_diffusion_model'))
    assert config['num_inference_timesteps']==4 and config['action_horizon']==40
    assert binding['model_files_sha256']['config.json']==donor['model_files_sha256']['config.json']
    assert set(original)==set(final) and len(final)==1030
    frozen=[];trainable=[]
    for name,value in final.items():
        assert value['dtype']==('BF16' if name.startswith('backbone.') else 'F32')
        assert all(value[k]==original[name][k] for k in ('dtype','shape','bytes'))
        if name.startswith(scope.FROZEN):
            assert value==original[name],('FROZEN_TENSOR_CHANGED',point,name);frozen.append(name)
        else:
            assert name.startswith('action_head.');trainable.append(dict(name=name,changed=value['sha256']!=original[name]['sha256']))
    assert len(frozen)==559 and len(trainable)==471 and any(r['changed'] for r in trainable)
    write(root/'evidence'/point/'FROZEN_SCOPE.json',dict(status='PASS_PI0267_SERIALIZED_PRECISION_AND_FROZEN_TENSOR_IDENTITY',
        model_binding=binding,donor_binding=donor,frozen=frozen,trainable=trainable,model_queries=0,gradient_queries=0))
    print(json.dumps(dict(status='PASS_FIXED_FINAL_MODEL_VIEW_AND_SCOPE',point=point,view=str(view))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('archive','materialize'));p.add_argument('--arm',choices=('N','R'),required=True)
    p.add_argument('--step',type=int,choices=(2048,4096));a=p.parse_args()
    if a.mode=='archive':assert a.step;archive(a.arm,a.step)
    else:assert a.step is None;materialize(a.arm)
