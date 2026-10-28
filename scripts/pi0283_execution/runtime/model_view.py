"""CPU exact master readback then native BF16 inference serialization."""
import argparse,hashlib,json,os,shutil
from pathlib import Path


def materialize(root,point,base,export):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from learning import file_digest
    from scope import digest
    root,base,export=map(Path,(root,base,export));torch.set_num_threads(4)
    receipt=json.loads((export/'EXPORT.json').read_text())
    assert receipt['status']=='PASS_CPU_EXACT_CHANGED_TENSOR_EXPORT_AND_FULL_STATE_HASH_PARITY'
    delta=export/receipt['exact_delta_file'];assert file_digest(delta)==receipt['exact_delta_sha256']
    output=root/'models'/point;output.mkdir(parents=True,exist_ok=True)
    done=output/'MATERIALIZATION.json'
    if done.exists():
        old=json.loads(done.read_text());assert old['export_receipt_sha256']==file_digest(export/'EXPORT.json')
        for name,sha in old['files'].items():assert file_digest(output/name)==sha
        return old
    original=json.loads((base/'model.safetensors.index.json').read_text());keys=original['weight_map']
    aliases=set(keys)-set(receipt['full_FP32_master_state'])
    assert aliases in (set(),{'backbone.model.lm_head.weight'}),('UNRECOGNIZED_SERIALIZED_ALIAS',aliases)
    alias_proof={}
    if aliases:
        alias='backbone.model.lm_head.weight';embedding='backbone.model.model.language_model.embed_tokens.weight'
        assert alias not in receipt['effective_bf16_tensor_sha256']
        with safe_open(base/keys[alias],framework='pt',device='cpu') as parent:
            value=parent.get_tensor(alias)
            assert dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))==receipt['full_FP32_master_state'][embedding]
        alias_proof=dict(omitted_native_tied_alias=alias,tied_parameter=embedding,pristine_alias_equals_frozen_master=True)
        keys.pop(alias)
    assert set(keys)==set(receipt['full_FP32_master_state'])
    expected=receipt['effective_bf16_tensor_sha256'];seen=set();files={};total_bytes=0
    with safe_open(delta,framework='pt',device='cpu') as changed:
        for shard in sorted(set(keys.values())):
            tensors={}
            with safe_open(base/shard,framework='pt',device='cpu') as pristine:
                for name in pristine.keys():
                    if name not in keys:continue
                    value=changed.get_tensor(name) if name in receipt['changed_tensors'] else pristine.get_tensor(name)
                    master=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=digest(value))
                    assert master==receipt['full_FP32_master_state'][name],('MASTER_RECONSTRUCTION',name)
                    quality=value.to(torch.bfloat16) if name in expected else value
                    if name in expected:assert digest(quality)==expected[name];seen.add(name)
                    tensors[name]=quality.contiguous()
                    total_bytes+=quality.numel()*quality.element_size()
            target=output/shard;temporary=target.with_suffix('.tmp')
            assert not target.exists(),'Audit a partial model view before recovery'
            save_file(tensors,str(temporary),metadata={'format':'pt'});temporary.replace(target)
            files[shard]=file_digest(target);del tensors
    assert seen==set(expected)
    for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
        assert file_digest(export/name)==receipt['model_files'][name]
        shutil.copyfile(export/name,output/name);files[name]=file_digest(output/name)
    original.setdefault('metadata',{})['total_size']=total_bytes
    (output/'model.safetensors.index.json').write_text(json.dumps(original,indent=2)+'\n')
    files['model.safetensors.index.json']=file_digest(output/'model.safetensors.index.json')
    result=dict(status='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW',point=point,export_receipt_sha256=file_digest(export/'EXPORT.json'),
        full_master_state_sha256=receipt['full_state_sha256'],files=files,effective_bf16_tensor_sha256=expected,model_queries=0,optimizer_transfer=False,alias_reconciliation=alias_proof)
    done.write_text(json.dumps(result,indent=2)+'\n')
    binding=dict(model_root=str(output),files=files,effective_bf16_tensor_sha256=expected)
    path=root/'design/MODEL_POINTS.json';points=json.loads(path.read_text()) if path.exists() else {}
    assert point not in points or points[point]==binding
    points[point]=binding;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(points,indent=2)+'\n');temporary.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','effective_bf16_tensor_sha256')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--point',required=True);parser.add_argument('--base',required=True);parser.add_argument('--export',required=True)
    arguments=parser.parse_args();materialize(arguments.root,arguments.point,arguments.base,arguments.export)
