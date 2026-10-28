"""Finite saved-window processor replay; no model, gradients or physics."""
import argparse
import json
from pathlib import Path
import numpy as np
from pi0266_audit import ROOT,BASE,DATA,read,sha,write,codec,build,batch,identity,instructions
from pi0264_plan import TEMPLATES


def main(output,compare):
    import torch
    torch.set_num_threads(1)
    stats=DATA/'dataset/statistics.json'
    assert sha(stats)==read(ROOT/'evidence/OBSERVATION_CODEC_AUDIT.json')['statistics_sha256']
    processor=build(read(stats),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    affine=codec.processor(read(stats))
    verdict=read(ROOT/'evidence/DIAGNOSTIC_VERDICT.json')
    families={f['family_id']:f for f in read(ROOT/'design/SCENE_CARD.json')['families']}
    rows=[]
    for parent in verdict['rows']:
        case=parent['result']['case']
        if case['expert']!='OBSERVABLE_CORRECTION_V1':continue
        windows=[w for w in read(Path(parent['attempt'])/'windows.json') if w['admitted']]
        assert windows
        for w in (windows[0],windows[-1]):
            assert sha(w['frame'])==w['frame_sha256'] and sha(w['proposal'])==w['proposal_sha256']
            f=dict(np.load(w['frame'],allow_pickle=False));p=dict(np.load(w['proposal'],allow_pickle=False))
            q=dict(np.load(Path(w['proposal']).parent/'codec.npz',allow_pickle=False))
            encoded=codec.encode(affine,p['state_v1'],p['raw_action'])
            for key,value in zip(('state','action','action_mask'),encoded):assert np.array_equal(q[key],value),(w['id'],key)
            if w['task']=='M1':
                language=TEMPLATES['T0'].format(color=families[w['family']]['factors']['colors'][w['command']])
            else:
                language=instructions('M3',w['command'],families[w['family']]['factors']['colors'])[0]
            packet={k:f[k] for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=language
            before=torch.get_rng_state().clone()
            free=batch(processor,packet)['inputs'];supervised=batch(processor,packet,p['raw_action'])['inputs']
            assert torch.equal(before,torch.get_rng_state()),'processor consumed Torch CPU RNG'
            assert identity(free)==identity({k:v for k,v in supervised.items() if k not in ('action','action_mask')})
            assert np.array_equal(supervised['state'].cpu().numpy(),q['state'][None])
            assert np.array_equal(supervised['action'].cpu().numpy(),q['action'][None])
            assert np.array_equal(supervised['action_mask'].cpu().numpy(),q['action_mask'][None])
            for key in ('time','phase','target_xyz','source_id','future_action'):
                try:batch(processor,dict(packet,**{key:1}))
                except ValueError:pass
                else:raise AssertionError(('INPUT_LEAK_ACCEPTED',key))
            rows.append(dict(id=w['id'],task=w['task'],command=w['command'],language=language,
                frame_sha256=w['frame_sha256'],proposal_sha256=w['proposal_sha256'],codec_sha256=sha(Path(w['proposal']).parent/'codec.npz'),
                condition_sha256=identity(free),labels_masks_sha256=identity({k:supervised[k] for k in ('action','action_mask')})))
    assert len(rows)==16
    receipt=dict(status='PASS_FINITE_ACTUAL_WINDOW_PROCESSOR_REPLAY',rows=rows,statistics_sha256=sha(stats),
        code_sha256=sha(__file__),free_supervised_condition_equal=True,stored_codec_exact=True,
        illegal_metadata_rejected=True,torch_CPU_RNG_unchanged=True,cross_process_equal=False,
        diagnostic_language_binding='M1 inherited T0; M3 inherited original first form; CPU replay only',
        diagnostic_windows_added_to_TRAIN=False,training_admission=False,model_queries=0,optimizer_updates=0,new_physics=0)
    if compare:
        previous=read(compare)
        assert previous['rows']==rows and previous['statistics_sha256']==receipt['statistics_sha256']
        receipt.update(cross_process_equal=True,previous_receipt_sha256=sha(compare))
    assert not output.exists(),'preserve previous replay receipt'
    write(output,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--compare',type=Path)
    args=parser.parse_args();main(args.output,args.compare)
