"""Freeze only this quality entry, with explicit reuse of closed old inputs."""
import importlib.util
from pathlib import Path
from infer_worker import ROOT,BASE,read,sha,write

def main():
    assert not (ROOT/'evidence/model_attempts.jsonl').exists()
    check=read(ROOT/'evidence/EVALUATION_CPU_CHECK.json')
    assert check['status']=='PASS_PI0269_ACTUAL_SHARED_ROT_INPUTS_AND_ORIGINAL_EXECUTOR_GATE' and len(check['checks'])==416
    assert check['source_virtual_sha256']==sha(ROOT/'code/virtual_episode.py')
    card=read(ROOT/'design/FIXED_OFFLINE_CARD.json');assert [len(card[k]) for k in ('fit_rows','factor_rows','legacy_rows')]==[224,304,440]
    assert card['logical_rows']==2904
    paths=[ROOT/'code'/name for name in ('infer_worker.py','offline.py','virtual_episode.py','component_episode.py','evaluate.py')]
    paths.extend(ROOT/'design'/name for name in ('EVALUATION_CARD.json','FIXED_OFFLINE_CARD.json','COMPONENT_CARD.json','PAIRING_PLAN.json'))
    paths.extend(ROOT/'evidence'/name for name in ('EVALUATION_CPU_CHECK.json','DONOR_EXACT_REUSE.json'))
    paths.append(ROOT/'contracts/ROT/source/dataset/statistics.json')
    old=read(BASE/'PI0268/R1/design/QUALITY_EXECUTION_FILESET.json')
    for row in old['files']:
        p=Path(row['path'])
        if not p.is_relative_to(BASE/'PI0268/R1'):
            assert sha(p)==row['sha256'],p
            paths.append(p)
    # Pure diagnostic visibility is the existing renderer function, not a new criterion.
    paths.append(BASE/'PI0267/code/pi0259_observation.py')
    for p in paths:assert p.is_file()
    write(ROOT/'design/QUALITY_EXECUTION_FILESET.json',dict(status='FROZEN_BEFORE_NEW_QUALITY',
        files=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))],
        reused_old_quality_seal_sha256=sha(BASE/'PI0268/R1/design/QUALITY_EXECUTION_FILESET.json'),
        new_offline=1936,new_main_episodes=128,component_proposals=96,model_queries_by_seal=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('FROZEN_PI0269_ACTUAL_QUALITY_FILESET',len(set(paths)),flush=True)

if __name__=='__main__':main()
