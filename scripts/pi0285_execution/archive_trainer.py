"""Reuse the mature full-Trainer archiver with a source-bound PI0285 root."""
import argparse,hashlib,json,os
from pathlib import Path


def archive(root,arm,step):
    root=Path(root);assert root.name=='PI0285' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    card=json.loads((root/'design'/(arm+'_CARD.json')).read_text())
    audit=root/'evidence'/f'{arm}-{step}_TRAINING_CPU.json'
    proof=json.loads(audit.read_text());assert proof['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    assert proof['point']==f'{arm}-{step}' and proof['checkpoint']==str(root/'learning'/arm/'main'/f'checkpoint-{step}')
    assert step<=card['updates'] and step in (1024,2048,4096)
    helper=root.parent/'PI0284/runtime/archive_trainer.py';source=helper.read_text()
    assert hashlib.sha256(helper.read_bytes()).hexdigest()=='126880ca6a0aab20ad556417335500f5b18f26c1f9fbbfdb81027b62049a064c'
    old='/scratch/e1192548/Projects/go2-arx5/experiments/PI0284'
    assert source.count(old)==3 and "assert len(entries) == 14" in source
    # ponytail: reuse upload/readback verbatim; only bind the new root and campaign.
    namespace={'__name__':'pi0284_mature_archive_bound_to_pi0285'}
    exec(compile(source.replace(old,str(root)),str(helper),'exec'),namespace)
    record=root/'archive';record.mkdir(exist_ok=True)
    binding=record/f'{arm}-{step}_HELPER_BINDING.json'
    value=dict(helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_CPU_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),root_substitution_only=True,campaign='PI0285',arm=arm,step=step,model_queries=0,optimizer_updates=0)
    if binding.exists():assert json.loads(binding.read_text())==value
    else:binding.write_text(json.dumps(value,indent=2)+'\n')
    namespace['main'](arm,step,max_steps=card['updates'],campaign='PI0285',token=(Path.home()/'.config/huggingface/token').read_text().strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();archive(a.root,a.arm,a.step)
