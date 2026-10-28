"""Actual hash/language overlap and loaded visibility source, from closed CPU arrays."""
import inspect,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274');OLD=ROOT.parent/'PI0273'
sys.path[:0]=[str(ROOT/'code'),str(OLD/'code'),str(ROOT.parent/'PI0267/code'),str(ROOT.parent/'PI0266/code'),str(ROOT.parent/'PI0250/runtime/reference-317a/src')]
from quality_gate import read,sha
from pi0255_motion3 import write
from pi0259_observation import visibility
def key(r):return r['condition_sha256'],r['labels_masks_sha256']
def main():
    card=read(ROOT/'design/FITTING_CARD.json');original=read(OLD/'design/FITTING_CARD.json');B={key(r) for r in original['base']};C={key(original['base'][i]) for i in original['target_source_indices']}
    trained=Counter(key(r) for r in card['streams']['EQUIL'] if r['stream']=='SUPPLEMENT');base=Counter(key(r) for r in card['base']);rows=[]
    for r in card['field_inputs']:
        rows.append(dict(r,exact_supplement_occurrences=trained[key(r)],exact_base_occurrences=base[key(r)],exact_historical_C=key(r) in C,exact_base_B=key(r) in B))
    assert len(rows)==1248 and sum(trained.values())==2048
    write(ROOT/'evidence/EXPOSURE_AND_ACTUAL_SEEN.json',dict(status='PASS_LITERAL_NEW_INPUT_AND_LABEL_AND_LANGUAGE_EXPOSURE',field_rows=rows,new_processed_conditions=len(card['new_inputs']),
        supplemental_unique_processed_pairs=len(trained),supplemental_unique_physical_poses=len({r['pose_id'] for r in card['streams']['EQUIL'] if r['stream']=='SUPPLEMENT'}),
        exact_field_inputs_consumed=sum(r['exact_supplement_occurrences']>0 for r in rows),exact_new_B_overlap=sum(key(r) in B for r in card['new_inputs']),exact_new_C_overlap=sum(key(r) in C for r in card['new_inputs']),
        field_pose_labels_are_TRAIN_diagnostics_not_blind_TEST=True,model_queries=0,teacher_calls=0,renders=0))
    source=Path(inspect.getfile(visibility));original_receipt=ROOT/'evidence/A0_ONSET_ANALYSIS.json'
    write(ROOT/'evidence/A0_VISIBILITY_SOURCE_BINDING.json',dict(status='PASS_ACTUAL_LOADED_VISIBILITY_SOURCE',actual_path=str(source),sha256=sha(source),
        previously_reported_path=read(original_receipt)['shared_visibility_source'],original_analysis_receipt_sha256=sha(original_receipt),numerical_tables_unchanged=True,new_model_queries=0,new_renders=0))
    print('PASS_LITERAL_EXPOSURE_AND_ACTUAL_VISIBILITY_SOURCE',len(trained),str(source),flush=True)
if __name__=='__main__':main()
