"""Reuse the sealed TASK8192 view for its newly authorized fixed FIT128 only."""
from pathlib import Path
from evaluation_card import ROOT,OLD,read,sha,freeze

def main():
    assert read(ROOT/'evidence/EVALUATION_CARD_FREEZE.json')['status']=='PASS_PREQUERY_FIXED_INPUT_NOISE_AND_REUSE_CARD'
    original=OLD/'model_bindings/TASK8192.json';binding=read(original)
    assert binding['source_revision']=='leoyuer/pi0246-gate-pass-campaign@7687ad4c04341b30c6a2a0959aeadf299fa3ba0e/arms/PI0276/TASK/checkpoint-8192'
    assert binding['checkpoint_tree_sha256']=='b78e300af57772c22ab8327918463afb1ea4b176689dbeba2ab243d4a27ae175'
    prior=read(OLD/'evidence/FINAL_POLICY_ADMISSION.json');assert prior['bindings']['TASK8192']==sha(original)
    for name,h in binding['model_files_sha256'].items():assert sha(Path(binding['view'])/name)==h
    freeze(ROOT/'model_bindings/TASK8192.json',binding)
    source_stats=OLD/'contracts/TASK/source/dataset/statistics.json';stats=ROOT/'contracts/TASK/source/dataset/statistics.json'
    stats.parent.mkdir(parents=True,exist_ok=True)
    if stats.exists():assert stats.read_bytes()==source_stats.read_bytes()
    else:stats.write_bytes(source_stats.read_bytes())
    assert sha(stats)==binding['statistics_sha256']=='b3da3d500d514ce3d1ce83001de30846d9ab76d699dc48831db8e4367ce98572'
    paths=[ROOT/'code'/name for name in ('infer.py','evaluate_fixed.py','evaluation_card.py','admit_donor_fit.py')]
    paths+=[ROOT/'design/EVALUATION_CARD.json',ROOT/'design/EVALUATION_NOISE.json',ROOT/'design/FIT_CARD.json',ROOT/'model_bindings/TASK8192.json',stats,original,OLD/'code/infer.py']
    # Retain the exact executed predecessor dependency index; its code and policy inputs
    # are reused, while PI0277 outputs and counters have separate roots.
    index=OLD/'design/EVALUATION_FILESET.json';paths.append(index)
    for r in read(index)['files']:
        p=Path(r['path'])
        if p.suffix=='.py' and '/PI0276/code/' not in str(p):assert sha(p)==r['sha256'];paths.append(p)
    freeze(ROOT/'design/EVALUATION_FILESET.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))],model_queries_before_seal=0,
        predecessor_infer_sha256=sha(OLD/'code/infer.py'),new_infer_sha256=sha(ROOT/'code/infer.py'),algorithm='UNCHANGED_STOCK_S4_BF16_AND_FP32_DECODE'))
    freeze(ROOT/'evidence/POLICY_ADMISSION_TASK8192.json',dict(status='PASS_PI0277_FIXED_POLICY_IDENTITY',point='TASK8192',allowed_lane='FIT',
        model_binding_sha256=sha(ROOT/'model_bindings/TASK8192.json'),evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),
        reuse_old_final_admission_sha256=sha(OLD/'evidence/FINAL_POLICY_ADMISSION.json'),new_model_queries=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('DONOR_FIT_IDENTITY_ADMITTED',flush=True)

if __name__=='__main__':main()
