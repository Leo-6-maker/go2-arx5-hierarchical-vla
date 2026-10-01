"""Read serialized checkpoint bytes; no model construction or GPU computation."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from pi0264_plan import read, write, sha

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0264')
DONOR=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0263/models/X2000/arms/PI0263/X/checkpoint-2000')
FROZEN=('backbone.','action_head.vlln.','action_head.vl_self_attention.')


def tensors(view,binding):
    for name,digest in binding['model_files_sha256'].items():
        assert sha(view/name)==digest
    entries={}
    for path in sorted(view.glob('*.safetensors')):
        with path.open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0]
            assert 0<length<16*2**20
            header=json.loads(stream.read(length))
            for name,meta in header.items():
                if name=='__metadata__':continue
                assert name not in entries
                start,end=meta['data_offsets'];assert 0<=start<end<=path.stat().st_size-8-length
                stream.seek(8+length+start);remaining=end-start;digest=hashlib.sha256()
                while remaining:
                    block=stream.read(min(remaining,8<<20));assert block
                    digest.update(block);remaining-=len(block)
                entries[name]=dict(dtype=meta['dtype'],shape=meta['shape'],bytes=end-start,sha256=digest.hexdigest())
    assert set(entries)==set(read(view/'model.safetensors.index.json')['weight_map'])
    return entries


def main(point):
    donor_binding=read(ROOT.with_name('PI0263')/'model_bindings/X2000.json')
    assert '@a2f9f767a11e4af269377603f3da432f4931d623/' in donor_binding['source_revision']
    binding=read(ROOT/'model_bindings'/f'{point}.json');view=Path(binding['view'])
    config=read(view/'config.json')
    assert all(config[k] is False for k in ('tune_llm','tune_visual','tune_vlln'))
    assert all(config[k] is True for k in ('tune_projector','tune_diffusion_model'))
    assert config['num_inference_timesteps']==4 and config['action_horizon']==40
    assert binding['model_files_sha256']['config.json']==donor_binding['model_files_sha256']['config.json']
    original,final=tensors(DONOR,donor_binding),tensors(view,binding)
    assert set(original)==set(final) and len(final)==1030
    frozen=[];trainable=[]
    for name,value in final.items():
        assert value['dtype']==('BF16' if name.startswith('backbone.') else 'F32')
        assert all(value[k]==original[name][k] for k in ('dtype','shape','bytes'))
        row=dict(name=name,**value)
        if name.startswith(FROZEN):
            assert value==original[name],('FROZEN_TENSOR_CHANGED',point,name)
            frozen.append(row)
        else:
            assert name.startswith('action_head.')
            trainable.append(dict(row,changed=value['sha256']!=original[name]['sha256']))
    assert len(frozen)==559 and len(trainable)==471 and any(r['changed'] for r in trainable)
    result=dict(status='PASS_PI0264_SERIALIZED_PRECISION_AND_FROZEN_TENSOR_IDENTITY',point=point,
        donor_binding=donor_binding,model_binding=binding,frozen_prefixes=FROZEN,frozen=frozen,trainable=trainable,
        scope_source='Reviewed Gr00tN1d7ActionHead.set_trainable_parameters; tune_vlln false freezes vlln/vl_self_attention; backbone tune_llm/tune_visual false',
        code_sha256=sha(__file__),model_queries=0,gradient_queries=0)
    write(ROOT/'evidence'/point/'FROZEN_SCOPE.json',result)
    print(json.dumps(dict(status=result['status'],point=point,frozen_tensors=len(frozen),
        trainable_tensors=len(trainable),changed_trainable=sum(r['changed'] for r in trainable))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--point',choices=('B2048','L2048'),required=True)
    main(parser.parse_args().point)
