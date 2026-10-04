"""Closed historical tables verified against their existing public readback receipts."""
import csv
import json
from pathlib import Path
from quality_gate import BASE,ROOT,read,sha
from prepare import write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5')
REVISIONS={'PI0270':'c62f4d010328c69f2b8f22fb91bb696cb3bd3539','PI0272':'4ae9b50af558fb438b1794e58e8bab592a11cf18'}

def table(campaign,name):
    receipt=read(BASE/campaign/'evidence/RESULT_ARCHIVE_summary.json')
    assert receipt['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS' and receipt['revision']==REVISIONS[campaign]
    row=next(r for r in receipt['files'] if r['path']==name)
    path=RAW/campaign/'delivery'/name
    assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    with path.open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if k=='source_row':continue
            if v in ('True','False'):row[k]=v=='True'
            elif v=='':row[k]=None
            else:
                try:row[k]=json.loads(v)
                except (ValueError,TypeError):pass
    return rows

def main():
    files={};counts={}
    for name,count in (('OFFLINE2904.csv',968),('NEW_FIT384.csv',192),('VIRTUAL192.csv',64),('COMPONENT96.csv',32)):
        rows=table('PI0270',name);own=[r for r in rows if r['point']=='AGG4096'];assert len(own)==count
        counts[name]=len(own);files[name]=sha(RAW/'PI0270/delivery'/name)
    rows=table('PI0272','ALL_POINTS3504.csv')
    for point in ('AGG4096','S4FIT1024'):
        assert len([r for r in rows if r['point']==point and r['panel']=='early_guard'])==192
        assert len([r for r in rows if r['point']==point and r['panel']=='support' and r['j'] in (0,1)])==592
    files['PI0272/ALL_POINTS3504.csv']=sha(RAW/'PI0272/delivery/ALL_POINTS3504.csv')
    write(ROOT/'evidence/HISTORICAL_TABLE_REUSE.json',dict(status='PASS_CLOSED_AGG_PANELS_AND_S4FIT_LOCAL_CONTEXT_ONLY',
        source_revisions=REVISIONS,files=files,AGG_counts=counts,S4FIT_main_outcomes='NOT_MEASURED_NOT_INVENTED',
        old_model_queries=0,repeated_historical_HF_readbacks=0,new_model_queries=0))
    print('PASS_CLOSED_AGG_PANELS_AND_S4FIT_LOCAL_CONTEXT_ONLY',flush=True)

if __name__=='__main__':main()
