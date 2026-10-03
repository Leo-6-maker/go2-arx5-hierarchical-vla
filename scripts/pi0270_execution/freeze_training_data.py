"""All fixed denominators first; then the sole authorized replacement mapping."""
import hashlib
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from admission import BASE, ROOT, read, sha, write
from plan import replacements, noise_seed
from infer_worker import build, batch, identity

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main():
    assert read(ROOT/'evidence/COLLECTION_PROCESS_COMPLETE.json')['status']=='ALL_FIXED_COLLECTION_FAMILY_PROCESSES_EXIT_ZERO'
    card=read(ROOT/'design/COLLECTION_CARD.json'); proposals=read(ROOT/'design/LABEL_PROPOSAL_CARD.json')['proposals']
    assert len(proposals)==6784
    records=[]; sources=[]; cells=defaultdict(set); reasons=Counter()
    for proposal in proposals:
        uid=proposal['uid']; path=RAW/'codec'/proposal['source_lane']/proposal['source_episode_key']/f"{proposal['query_index']:03d}.json"
        row=read(path); assert row['source']['source_lane']==proposal['source_lane']
        assert (row['source']['family'],row['source']['task'],row['source']['command'],row['source']['query_index'])==(proposal['family'],proposal['task'],proposal['command'],proposal['query_index'])
        sources.append(dict(uid=uid,path=str(path),sha256=sha(path)))
        if row['eligible']:
            assert row['roundtrip']['PASS'] and row['train_eval_bit_identity'] and row['collector_full_condition_bit_identity']
            cells[proposal['family'],proposal['task'],proposal['command']].add(row['processed_state_sha256'])
            row['uid']=uid;row['codec_path']=str(path);row['codec_sha256']=sha(path)
        reasons.update(row.get('reasons',row['source'].get('reasons',[])))
        records.append(row)
    families=list(dict.fromkeys(c['family'] for c in card['cases']))
    coverage=[dict(family=f,task=t,command=c,eligible_distinct_processed_states=len(cells[f,t,c]),PASS=len(cells[f,t,c])>=8)
              for f in families for t in ('M1','M3') for c in (0,1)]
    assert len(coverage)==96
    passed=all(r['PASS'] for r in coverage)
    write(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json',dict(status='PASS_ALL96_FIXED_NEW_SUPPORT_CELLS' if passed else 'PI0270_DATA_COVERAGE_INSUFFICIENT',
          coverage=coverage,proposals=6784,eligible=sum(r['eligible'] for r in records),excluded_or_missing=sum(not r['eligible'] for r in records),
          exclusion_counts=dict(reasons),all_fixed_denominators_preserved=True,source_records=sources,
          new_model_optimizer_queries_by_admission=0,statistics_refits=0,source_sha256=sha(__file__)))
    if not passed:
        print('PI0270_DATA_COVERAGE_INSUFFICIENT',sum(not r['PASS'] for r in coverage),flush=True);return
    assert sha(ROOT/'design/PAIRING_PLAN.json')=='fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'
    plan=read(ROOT/'design/PAIRING_PLAN.json'); picked=replacements(plan,records)
    assert len(picked)==4096 and len(plan['units'])==4096
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original_call=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original_call(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    torch.nn.Module.__call__=guard
    def forbidden(*a,**kw):raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json'
    proc=build(read(stats),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    mappings=[]; used=Counter(); fit=[]; seen=defaultdict(set); counts=Counter()
    for selection in picked:
        source=selection['source']; original=source['source']; unit=plan['units'][selection['unit']-1]
        assert unit['slot']=='RECOVERY_INTERVENTION' and unit['order'][selection['micro']]==selection['command']
        text=plan['templates'][unit['template']].format(color=plan['source_family_colors'][unit['family']][selection['command']]) if unit['task']=='M1' else source['language']
        assert sha(original['sample'])==original['sample_sha256']
        with np.load(original['sample']) as z:
            packet=dict(body_rgb=z['body_rgb'].copy(),wrist_rgb=z['wrist_rgb'].copy(),state_v1=z['state_v1'].copy(),language=text)
            raw=z['raw_action'].copy()
        inputs=batch(proc,packet,raw)['inputs']; condition={k:v for k,v in inputs.items() if k not in ('action','action_mask')}
        pure=batch(proc,packet)['inputs']; assert identity(condition)==identity(pure)
        condition_sha=identity(condition); label_sha=identity({k:inputs[k] for k in ('action','action_mask')})
        row=dict(unit=unit['unit'],micro=selection['micro'],family=unit['family'],task=unit['task'],command=selection['command'],
                 template=unit['template'],language=text,bucket=selection['bucket'],source_uid=source['uid'],
                 source_codec_path=source['codec_path'],source_codec_sha256=source['codec_sha256'],
                 sample=original['sample'],sample_sha256=original['sample_sha256'],condition_sha256=condition_sha,
                 labels_masks_sha256=label_sha,processed_state_sha256=source['processed_state_sha256'],
                 collector_language=source['language'],same_collector_text=text==source['language'],
                 scheduled_original_anchor_provenance=unit['anchor'],scheduled_original_variant_provenance=unit['recovery_variant'],
                 new_actual_projection_ell_m=original['teacher']['ell_m'],projected_terminal=original['teacher']['terminal_hold'],
                 zero_teacher_prefix_displacement=original['diagnostic']['hold_case'])
        if row['same_collector_text']:assert condition_sha==source['condition_sha256']
        mappings.append(row);used[source['uid']]+=1
        counts[selection['bucket']]+=1;counts['projected_terminal']+=int(row['projected_terminal'])
        counts['zero_teacher_prefix_displacement']+=int(row['zero_teacher_prefix_displacement'])
        cell=unit['family'],unit['task'],selection['command']
        if len(seen[cell])<2 and condition_sha not in seen[cell]:
            seen[cell].add(condition_sha);input_id=f'{cell[0]}-{cell[1]}-c{cell[2]}-r{len(seen[cell])-1}'
            namespace=f'PI0270:NEWFIT:27102:{input_id}';seed=noise_seed(namespace)
            tensor=torch.randn((1,40,132),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
            noise_file=RAW/'noise-newfit'/(input_id+'.npz');noise_file.parent.mkdir(parents=True,exist_ok=True)
            if noise_file.exists():
                with np.load(noise_file) as z:assert np.array_equal(z['float32'],tensor.numpy())
            else:np.savez_compressed(noise_file,float32=tensor.numpy())
            fit.append(dict(input_id=input_id,source_mapping=row,input=original['sample'],input_sha256=original['sample_sha256'],
                            language=text,condition_sha256=condition_sha,labels_masks_sha256=label_sha,
                            noise_file=str(noise_file),noise_file_sha256=sha(noise_file),namespace=namespace,seed=seed,
                            noise_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),actual_consumption='NOT_YET_TRAINED_MUST_PROVE_AFTER_MAIN'))
    fit_pass=len(fit)==192 and len(seen)==96 and all(len(v)==2 for v in seen.values())
    write(ROOT/'evidence/NEWFIT_ADMISSION.json',dict(status='PASS_FIXED_NEWFIT192_AVAILABLE' if fit_pass else 'PI0270_DATA_COVERAGE_INSUFFICIENT',
          rows=len(fit),cell_condition_counts={str(k):len(v) for k,v in seen.items()},required_rows=192,
          no_replacement_sampling=True,model_queries=0,optimizer_updates=0))
    if not fit_pass:
        print('PI0270_DATA_COVERAGE_INSUFFICIENT_FIXED_NEWFIT',len(fit),flush=True);return
    write(ROOT/'design/RECOVERY_MAPPING.json',dict(schema='PI0270_UNIQUE_FIXED_RECOVERY_MAPPING_V1',rows=mappings,
          common_units=2048,common_microbatches=4096,recovery_units=2048,recovery_microbatches=4096,
          original_plan_sha256=sha(ROOT/'design/PAIRING_PLAN.json'),data_admission_sha256=sha(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json'),
          counts=dict(counts),used_proposals=dict(used),unused_proposals=[p['uid'] for p in proposals if p['uid'] not in used],
          old_support_all_seen_claim=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    write(ROOT/'design/NEW_FIT192.json',dict(schema='PI0270_FIXED_NEW_FIT192_V1',rows=fit,points=['PREFIX4096','AGG4096'],
          fixed_queries=384,TRAIN_support_diagnosis=True,blind_TEST=False,queries_allowed_after_AGG_final_only=True,
          statistics_sha256=sha(stats),source_mapping_sha256=sha(ROOT/'design/RECOVERY_MAPPING.json')))
    print('PASS_FIXED_4096_RECOVERY_MICROS_AND_NEW_FIT192',flush=True)


if __name__=='__main__':main()
