"""Join completed optimizer-prefix consumption to frozen TRAIN support on CPU."""
import argparse,collections,json
from pathlib import Path
from plan import read,sha,write,exposure

def main(root,journal,step):
    membership=read(root/'design/ELIGIBLE_MEMBERSHIP.json');rows=membership['rows'];proof={r['uid']:r for r in rows}
    groups=collections.defaultdict(list)
    for row in rows:groups[row['seed'],row['group']].append(row)
    groups={k:sorted(v,key=lambda r:(r['anchor'],r['uid'])) for k,v in groups.items()}
    plan=read(root/'design/TRAIN_PLAN.json')['stream'];actual=[json.loads(s) for s in journal.read_text().splitlines()][:2*step]
    assert len(actual)==2*step and [r['micro_index'] for r in actual]==list(range(2*step))
    for expected,observed in zip(plan,actual):
        assert all(observed[k]==v for k,v in expected.items()),(expected,observed)
        row=proof[observed['uid']]
        for k in ('condition_sha256','labels_masks_sha256','group','seed','anchor'):assert observed[k]==row[k],k
    write(root/'evidence'/f'ACTUAL_EXPOSURE_{step}.json',dict(status='PASS_ACTUAL_COMPLETED_PREFIX_AND_FROZEN_PLAN',completed_update=step,consumed_microbatches=len(actual),
        source_journal=str(journal),source_journal_sha256=sha(journal),plan_sha256=sha(root/'design/TRAIN_PLAN.json'),rows=exposure(actual,groups,proof),
        prefetch_is_not_consumption=True,full4096_coverage_claimed=step==4096,new_model_queries=0,new_physics=0))
    print('PASS_ACTUAL_COMPLETED_PREFIX',step,len(actual))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('journal',type=Path);p.add_argument('step',type=int,choices=(512,1024,2048,4096));a=p.parse_args();main(a.root,a.journal,a.step)
