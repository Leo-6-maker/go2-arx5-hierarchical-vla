"""Reuse closed AGG/BROAD tables through their fixed-revision member receipts."""
import importlib.util
from quality_gate import BASE,ROOT,RAW,read,sha
from quality_worker import write
source=BASE/'PI0273/code/reuse.py'
spec=importlib.util.spec_from_file_location('pi0273_closed_table_reader',source)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.ROOT=ROOT
previous.REVISIONS['PI0273']='5cd59ada4e39eaeac57734817f74ee58207a1acd'
table=previous.table

def main():
    checks=[]
    for name,count in (('OFFLINE2904.csv',968),('NEW_FIT576.csv',192),('VIRTUAL192.csv',64),('COMPONENT96.csv',32)):
        rows=table('PI0273',name)
        for point in ('AGG4096','BROAD5120'):assert len([r for r in rows if r['point']==point])==count
        checks.append(dict(name=name,sha256=sha(RAW.parent/'PI0273/delivery'/name)))
    for name,panel,count in (('EARLY_CONTEXT768.csv','early_guard',192),('LATE_CONTEXT2368.csv','support',592)):
        rows=table('PI0273',name)
        for point in ('AGG4096','BROAD5120','S4FIT1024'):
            assert len([r for r in rows if r['point']==point and r['panel']==panel])==count
        checks.append(dict(name=name,sha256=sha(RAW.parent/'PI0273/delivery'/name)))
    write(ROOT/'evidence/HISTORICAL_TABLE_REUSE.json',dict(status='PASS_CLOSED_AGG_BROAD_ALL_ORIGINAL_PANELS_AND_S4FIT_CONTEXT',
        source_revision=previous.REVISIONS['PI0273'],files=checks,new_model_queries=0,repeated_historical_readbacks=0))
    print('PASS_CLOSED_AGG_BROAD_PANELS_NO_REPEAT_QUERY',flush=True)
if __name__=='__main__':main()
