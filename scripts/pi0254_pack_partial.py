"""Pack fixed completed evidence; no models, inference or training."""
from pathlib import Path
import json
import hashlib
import tarfile

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0254')


def main():
    report=json.loads((ROOT/'partial_closeout_with_raw_counters.json').read_text())
    assert report['query_attempts']==1088 and report['inference_head_forwards']==4352
    assert report['raw_training_counters_including_engineering']['raw_U576_head_forwards']==41477
    files=list((ROOT/'eval/U576_4608').rglob('*'))
    files += list((ROOT/'evidence/training_closeout').glob('*'))
    files += [ROOT/n for n in ('partial_closeout.json','partial_closeout_with_raw_counters.json','query_attempts.jsonl','head_forwards.jsonl','eval_card.json',
                              'U576_4608_evaluation_intent.json','U_FINAL_REPORT.md')]
    files += [ROOT/'execution'/n for n in ('pi0254_partial_closeout.py','pi0254_pack_partial.py','pi0254_eval.py','pi0254_compare.py')]
    files=sorted({p for p in files if p.is_file()})
    assert all(not p.is_symlink() and p.resolve().is_relative_to(ROOT) for p in files)
    assert all(p.suffix in ('.json','.jsonl','.npz','.py','.md','.gz') for p in files)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    out=ROOT/'evidence/U_final_partial_payload';out.mkdir(exist_ok=True)
    (out/'MEMBER_MANIFEST.json').write_text(json.dumps(dict(files=entries),indent=2)+'\n')
    with tarfile.open(out/'PI0254_U_final_partial.tar.gz','w:gz') as tar:
        for path in files:tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print(json.dumps(dict(files=len(entries),bytes=sum(x['bytes'] for x in entries))),flush=True)


if __name__=='__main__':main()
