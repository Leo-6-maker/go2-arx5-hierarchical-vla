"""Finite CPU/reset dispatch using the already accepted native environment."""
import json
import subprocess
import sys
import numpy as np
from pi0266_common import ROOT,BASE,RAW,read,write,sha
sys.path.insert(0,str(BASE/'PI0256/runtime'))
import pi0255_run as previous


def environment():
    previous.ROOT=ROOT
    py,env=previous.environment();env['PYTHONPATH']+=':'+str(ROOT/'code')
    return py,env


def seal():
    assert not (ROOT/'evidence/RAW_BUDGET.json').exists()
    for name in ('OBSERVATION_CODEC_AUDIT.json','EXPERT_CPU.json','PATH_V2_CPU.json','NEUTRAL_NATIVE_CPU.json'):
        assert read(ROOT/'evidence'/name)['status'].startswith('PASS')
    files={str(p.relative_to(ROOT)):sha(p) for folder in ('code','design','evidence') for p in (ROOT/folder).glob('*') if p.is_file() and p.name!='RESET_EXECUTION_SEAL.json'}
    write(ROOT/'design/RESET_EXECUTION_SEAL.json',dict(authority=read(ROOT/'code/PUBLICATION_RECEIPT.json'),files=files,physics_starts=0))
    print('PASS_RESET_SOURCE_AND_CPU_ADMISSION_SEALED',flush=True)


def resets():
    py,env=environment();rows=[]
    for case in read(ROOT/'design/SCENE_CARD.json')['reset_cases']:
        folder=RAW/'resets'/case['case_id'];prior=list(folder.glob('attempt-*/result.json'))
        valid=[p for p in prior if read(p)['validity']=='VALID']
        if not valid:
            log=ROOT/'logs'/(case['case_id']+'-%02d.log'%len(prior))
            with log.open('wb') as stream:
                done=subprocess.run([str(py),str(ROOT/'code/pi0266_reset.py'),'--case',case['case_id']],env=env,stdout=stream,stderr=subprocess.STDOUT)
            if done.returncode:raise SystemExit(done.returncode)
            valid=[p for p in folder.glob('attempt-*/result.json') if read(p)['validity']=='VALID']
        assert len(valid)==1
        row=read(valid[0]);rows.append(dict(result=row,attempt=str(valid[0].parent)))
        print(json.dumps(dict(case=case['case_id'],safe=row['safe'],failure=row.get('failure'))),flush=True)
    states=[dict(np.load(row['attempt']+'/reset_state.npz',allow_pickle=False)) for row in rows]
    differences=[]
    for a in states[1:]:
        first=states[0];physical_keys=['root','dof','TCP','rotation']
        for key in physical_keys+rows[0]['result']['history_fields']:
            error=float(np.max(abs(a[key]-first[key])));differences.append(dict(key=key,max_abs=error,bitwise_equal=np.array_equal(a[key],first[key])))
    reproducible=all(x['max_abs']<=1e-5 for x in differences)
    safe=all(r['result']['safe'] for r in rows)
    status='PASS_FIXED_RESET_SAFE_CONTINUE_TEACHER_DIAGNOSTICS' if safe and reproducible else 'FIXED_RESET_NOT_REPRODUCIBLY_SAFE'
    write(ROOT/'evidence/FIXED_RESET_VERDICT.json',dict(status=status,overall='IN_PROGRESS' if safe and reproducible else 'PARTIAL',cases=rows,
        reproducible=reproducible,all_safe=safe,differences=differences,numerical_tolerance=1e-5,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(status=status,safe=safe,reproducible=reproducible)),flush=True)


if __name__=='__main__':
    if '--seal' in sys.argv:seal()
    elif '--resets' in sys.argv:resets()
    else:
        py,env=environment();raise SystemExit(subprocess.run([str(py),str(ROOT/'code/pi0266_reset.py')]+sys.argv[1:],env=env).returncode)
