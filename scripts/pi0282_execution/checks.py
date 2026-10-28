"""Synthetic CPU serialization/metric fixtures, never research observations."""
import argparse,json,tempfile
from pathlib import Path


def run(root):
    import numpy as np,torch
    from safetensors.torch import save_file
    from learning import export_point,file_digest
    from model_view import materialize
    from evaluate import recompute
    torch.set_num_threads(4);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-cpu-',dir=root) as directory:
        r=Path(directory);base=r/'base';checkpoint=r/'checkpoint';export=r/'export'
        for p in (base,checkpoint,r/'design'):p.mkdir()
        key='action_head.action_decoder.synthetic';original={key:torch.zeros(32,2,dtype=torch.bfloat16),'backbone.synthetic':torch.ones(2,dtype=torch.bfloat16)}
        save_file(original,str(base/'model.safetensors'))
        (base/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={k:'model.safetensors' for k in original})))
        master={k:v.clone() for k,v in original.items()};master[key]=master[key].float();master[key][10]=.125
        save_file(master,str(checkpoint/'model.safetensors'))
        for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):(checkpoint/name).write_text('{}\n')
        receipt=export_point(checkpoint,base,export,set(original))
        for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):(export/name).write_bytes((checkpoint/name).read_bytes())
        view=materialize(r,'SYNTHETIC',base,export);assert view['full_master_state_sha256']==receipt['full_state_sha256']
        folder=r/'source/diagnostics/SYNTHETIC';folder.mkdir(parents=True);rows=[]
        target=np.tile(np.array([0,0,0,1,0,0,0,1,0,1],np.float32),(40,1))
        for i in range(48):
            p=folder/f'{i:02d}';p.mkdir();np.savez(p/'output.npz',decoded=target,normalized_OOB_fraction=0.);np.savez(p/'target.npz',commands=target)
            rows.append(dict(index=i,uid='SYNTHETIC-'+str(i),split='TRAIN' if i<24 else 'VAL',phase=i%4,output_sha256=file_digest(p/'output.npz'),target_sha256=file_digest(p/'target.npz')))
        (folder/'RESULT.json').write_text(json.dumps(dict(rows=rows)))
        result=recompute(r,'SYNTHETIC');assert result['fit_adequate'] and result['TRAIN_fit_count']==24
        for i in (0,4,8,12):
            p=folder/f'{i:02d}';invalid=target.copy();invalid[:]=np.nan;np.savez(p/'output.npz',decoded=invalid,normalized_OOB_fraction=0.);rows[i]['output_sha256']=file_digest(p/'output.npz')
        (folder/'RESULT.json').write_text(json.dumps(dict(rows=rows)))
        result=recompute(r,'SYNTHETIC');assert not result['fit_adequate'] and result['TRAIN_fit_count']==20
    print('PASS_SYNTHETIC_EXACT_MASTER_TRANSFER_BF16_VIEW_PREFIX_AND_PHASE_GATES; neural/optimizer/physics=0')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);arguments=parser.parse_args();run(arguments.root)
