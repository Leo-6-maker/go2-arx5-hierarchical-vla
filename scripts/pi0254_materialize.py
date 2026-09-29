"""Materialize only the four predeclared PI0254 model views from full HF receipts."""
from pathlib import Path
import argparse
import hashlib
import os
import sys

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0254')
sys.path.insert(0,str(ROOT.parent/'PI0253/runtime'))
import pi0253_full_screen_v2 as inherited


def bind(point):
    arm,step=point.split('_');step=int(step)
    assert arm in ('U576','R576') and step in (1152,4608)
    card_path=ROOT/f'bindings/{arm}_run_card.json';card=inherited.read(card_path)
    expected={'U576':'75b8c857bcbb1daabc087c1d9df2eafc72de76ca1c21ab65f8c8ec9c7715bb95',
              'R576':'4bdc9aaf499f6da9a0590f1600b02500a2a199ce7a9f944b087d00e20ec9a4dd'}
    assert inherited.sha(card_path)==expected[arm] and card['stage']['evaluation_steps']==[1152,4608]
    durable=inherited.read(ROOT/f'bindings/{point}.durable_receipt.json')
    readback=inherited.read(ROOT/f'bindings/{point}.full_content_readback.json')
    assert durable['status']=='DURABLE_VERIFIED' and durable['family']==card['family'] and durable['checkpoint_step']==step
    record=durable['source_record'];inventory=record['inventory']
    assert {'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin'}<={r['path'] for r in inventory['entries']}
    assert record['checkpoint_step']==step and record['max_steps']==4608 and record['family']==card['family']
    assert readback['status']=='PASS_PT_FULL_HF_READBACK' and readback['full_content_stream_readback']
    assert readback['revision']==durable['hf_manifest_commit_id'] and readback['source_tree_sha256']==inventory['tree_sha256']
    assert readback['repo_id']==card['archive']['repo_id'] and readback['family']==card['family']
    prefix=readback['path_prefix'];base=card['archive']['path_prefix']
    assert prefix==base or prefix.startswith(base+'/recovery/')
    observed={row['path']:row for row in readback['files']}
    assert set(observed)=={prefix+f'/checkpoint-{step}/'+row['path'] for row in inventory['entries']}
    for row in inventory['entries']:
        actual=observed[prefix+f'/checkpoint-{step}/'+row['path']]
        assert actual['bytes']==row['bytes'] and actual['sha256']==row['sha256']
    entries=[row for row in inventory['entries'] if row['path'] in {'config.json','model.safetensors.index.json',
                 'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'}]
    assert len(entries)==4
    return card,durable,readback,entries


def materialize(point):
    import requests
    from huggingface_hub import hf_hub_url
    card,durable,readback,entries=bind(point)
    step=durable['checkpoint_step'];view=ROOT/'models'/point;view.mkdir(parents=True,exist_ok=True)
    prefix=readback['path_prefix']+f'/checkpoint-{step}'
    for row in entries:
        path=view/row['path']
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial');digest=hashlib.sha256();size=0
            with requests.get(hf_hub_url(readback['repo_id'],filename=prefix+'/'+row['path'],revision=readback['revision']),
                              stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(8<<20):stream.write(block);digest.update(block);size+=len(block)
                    stream.flush();os.fsync(stream.fileno())
            assert size==row['bytes'] and digest.hexdigest()==row['sha256'];os.replace(temporary,path)
        assert path.stat().st_size==row['bytes'] and inherited.sha(path)==row['sha256']
        print('PINNED_PI0254_MODEL_FILE',point,row['path'],flush=True)
    restore=dict(schema='PI0254_PINNED_MODEL_ONLY_V1',checkpoint_step=step,family=card['family'],
                 optimizer_restored=False,repo_id=readback['repo_id'],revision=readback['revision'],prefix=prefix,
                 checkpoint_tree_sha256=readback['source_tree_sha256'],files=entries)
    inherited.write(view/'pi0245_inference_restore.json',restore)
    inherited.write(ROOT/f'models/{point}.binding.json',dict(durable_receipt=durable,full_content_readback=readback,
                 source_run_card_sha256=inherited.sha(ROOT/f'bindings/{card["arm"]}_run_card.json')))
    print('PASS_MODEL_ONLY_MATERIALIZATION_NOT_QUERIED',point,flush=True)


def check():
    import copy
    from unittest.mock import patch
    cards={arm:inherited.read(ROOT/f'bindings/{arm}_run_card.json') for arm in ('U576','R576')}
    original=inherited.read
    names=['config.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors',
           'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']
    fixtures={}
    for arm,card in cards.items():
        for step in (1152,4608):
            point=f'{arm}_{step}';prefix=card['archive']['path_prefix'];entries=[dict(path=n,bytes=1,sha256='0'*64) for n in names]
            durable=dict(status='DURABLE_VERIFIED',family=card['family'],checkpoint_step=step,hf_manifest_commit_id='CPU-fixture',
                         source_record=dict(family=card['family'],checkpoint_step=step,max_steps=4608,inventory=dict(tree_sha256='CPU-fixture-tree',entries=entries)))
            readback=dict(status='PASS_PT_FULL_HF_READBACK',full_content_stream_readback=True,revision='CPU-fixture',
                          source_tree_sha256='CPU-fixture-tree',family=card['family'],repo_id=card['archive']['repo_id'],path_prefix=prefix,
                          files=[dict(row,path=prefix+f'/checkpoint-{step}/'+row['path']) for row in entries])
            fixtures[ROOT/f'bindings/{point}.durable_receipt.json']=durable
            fixtures[ROOT/f'bindings/{point}.full_content_readback.json']=readback
    def read(path):return fixtures[path] if path in fixtures else original(path)
    with patch.object(inherited,'read',side_effect=read):
        for point in ('U576_1152','U576_4608','R576_1152','R576_4608'):assert len(bind(point)[3])==4
        path=ROOT/'bindings/U576_1152.full_content_readback.json';saved=copy.deepcopy(fixtures[path])
        for key,value in [('full_content_stream_readback',False),('revision','wrong'),('family','wrong'),('path_prefix','wrong')]:
            fixtures[path]={**saved,key:value}
            try:bind('U576_1152')
            except AssertionError:pass
            else:raise AssertionError('poisoned materialization identity accepted: '+key)
        fixtures[path]=saved
    inherited.write(ROOT/'materialization_cpu_receipt.json',dict(status='PASS_FOUR_FIXED_POINT_BINDINGS_CPU',
          payload='SYNTHETIC_RECEIPTS_REAL_FROZEN_CARDS',code_sha256=inherited.sha(Path(__file__)),new_model_queries=0,new_optimizer_updates=0))
    print('PASS_FOUR_FIXED_POINT_BINDINGS_CPU',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--point',choices=[f'{a}_{s}' for a in ('U576','R576') for s in (1152,4608)])
    parser.add_argument('--cpu-check',action='store_true');args=parser.parse_args()
    if args.cpu_check:check()
    elif args.point:materialize(args.point)
    else:parser.error('--point required')
