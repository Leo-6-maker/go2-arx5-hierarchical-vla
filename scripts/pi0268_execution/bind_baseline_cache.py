"""Exact OLD0 cache provenance and two effective step-zero policy bindings."""
import hashlib
import json
from pathlib import Path
import numpy as np
from freeze_contracts import ROOT,OLD,RAW,read,sha,write

def main():
    card=read(ROOT/'design/APPROVED_RUN_CARD.json');binding=read(OLD/'R1/model_bindings/donor.json')
    assert binding['checkpoint_tree_sha256']==card['donor']['tree_sha256']
    assert binding['source_revision']==card['donor']['repo']+'@'+card['donor']['revision']+'/'+card['donor']['prefix']
    assert sha(OLD/'R1/design/statistics.json')==card['contracts']['OLD']
    prior_card=read(OLD/'R1/design/EVALUATION_CARD.json');current=read(ROOT/'design/EVALUATION_CARD.json')
    for k in ('dt_s','executors','generator','legacy_scorer','max_queries_per_episode','noise','replan_s','scorer_sha256','source_scene_sha256','tool_card_sha256'):
        assert prior_card[k]==current[k],k
    chain=OLD/'R1/evidence/QUERY_CHAIN_REPLAY_donor.json';chain_receipt=read(chain)
    assert chain_receipt['status']=='PASS_SAVED_OBSERVATION_NOISE_AND_EXECUTION_CHAIN' and chain_receipt['queries']==6400
    records=sorted((RAW/'virtual/donor').glob('*/result.json'));assert len(records)==64
    virtual=[]
    for p in records:
        rec=read(p);assert rec['validity']=='VALID' and rec['point']=='donor'
        assert sha(p.parent/'trajectory.npz')==rec['arrays_sha256']
        assert rec['evaluation_card_sha256']==sha(OLD/'R1/design/EVALUATION_CARD.json')
        for receipt in (p.parent/'queries').glob('*/prediction.receipt.json'):
            r=read(receipt);assert r['binding']==binding and r['native_S4'] and r['heads']==4
        virtual.append(dict(case=p.parent.name,path=str(p),record_sha256=sha(p),arrays_sha256=rec['arrays_sha256']))
    legacy=[]
    for row in read(ROOT/'design/FIXED_OFFLINE_CARD.json')['legacy_rows']:
        p=RAW/'offline/donor/legacy'/row['uid']/'record.json';r=read(p);assert r['row']==row and r['status']=='VALID'
        original=Path(row['PI0267_donor_cell'])/'record.json';old=read(original)
        assert sha(original)==row['PI0267_donor_record_sha256']
        for key in ('model_files_sha256','checkpoint_tree_sha256','source_revision'):assert old['model_binding'][key]==binding[key]
        tensor=Path(r['tensor']);assert sha(tensor)==r['tensor_sha256']==row['PI0267_donor_tensor_sha256']
        with np.load(tensor) as z:
            assert hashlib.sha256(z['noise'].tobytes()).hexdigest()==row['identities']['noise_sha256']
            with np.load(RAW/'noise'/('legacy-'+row['noise']+'.npz')) as nz:assert np.array_equal(z['noise'],nz['float32'])
        legacy.append(dict(uid=row['uid'],record=str(p),record_sha256=sha(p),tensor=str(tensor),tensor_sha256=r['tensor_sha256']))
    assert len(legacy)==440
    for point,arm in [('OLD0','OLD'),('ROT0','ROT')]:
        write(ROOT/'model_bindings'/(point+'.json'),dict(binding,point=point,schema='PI0268_R1_EFFECTIVE_POLICY_MODEL_ONLY_V1',statistics_sha256=card['contracts'][arm],
            effective_contract=arm,distinct_effective_policy_even_with_same_weights=True))
    write(ROOT/'evidence/OLD0_EXACT_CACHE_BINDING.json',dict(status='PASS_EXACT_OLD0_MODEL_PROCESSOR_INPUT_NOISE_CONVENTIONS',
        donor_binding=binding,old_statistics_sha256=card['contracts']['OLD'],old_virtual_card_sha256=sha(OLD/'R1/design/EVALUATION_CARD.json'),
        new_virtual_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),reused_query_chain_receipt_sha256=sha(chain),
        virtual=virtual,legacy=legacy,model_queries=0,renderer_calls=0,physics_steps=0,ROT0_reuse_forbidden=True))
    print('PASS_EXACT_OLD0_CACHE',len(virtual),len(legacy),flush=True)

if __name__=='__main__':main()
