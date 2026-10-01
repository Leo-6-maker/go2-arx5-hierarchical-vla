#!/usr/bin/env python3
"""Read-only housekeeping regression check; never performs GitHub writes.

Compares the unchanged legacy CPU suite at the accepted pre-maintenance commit
with the proposed housekeeping commit in the same environment. Does not hide
legacy failures, skip new failures, or modify scientific code to make CI green.
"""
from pathlib import Path
import json, os, re, subprocess, sys, tempfile, xml.etree.ElementTree as ET
BASE = '607c7b734ba88ee0215a432aa8d2f576da9e77b0'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'repo-admin-validation'
ALLOWED = {'README.md', 'docs/PROJECT_STATUS.md','docs/RESEARCH_INDEX.md',
           'docs/REPOSITORY_WORKFLOW.md','.github/workflows/repository-admin-checks.yml'}
def cmd(args, cwd=ROOT, check=True, env=None):
    return subprocess.run(args,cwd=cwd,env=env,text=True,capture_output=True,check=check,timeout=180)
def main():
    OUT.mkdir(exist_ok=True)
    paths=cmd(['git','diff','--name-only',BASE,'HEAD']).stdout.splitlines()
    forbidden=[p for p in paths if p not in ALLOWED and not p.startswith(('docs/maintenance/2026-10-01/','scripts/maintenance/'))]
    if forbidden: raise RuntimeError('Unexpected non-administrative change: '+str(forbidden))
    missing=[]
    for rel in ['README.md','docs/PROJECT_STATUS.md','docs/RESEARCH_INDEX.md','docs/REPOSITORY_WORKFLOW.md']:
        p=ROOT/rel
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'): continue
            q=target.split('#',1)[0]
            if q and not (p.parent/q).exists():missing.append([rel,target])
    if missing: raise RuntimeError('Broken local navigation links: '+str(missing))
    tests=cmd([sys.executable,str(ROOT/'scripts/maintenance/test_consolidate_repo.py')])
    (OUT/'administrative-tests.txt').write_text(tests.stdout+tests.stderr)
    results={}
    with tempfile.TemporaryDirectory(prefix='pre-consolidation-') as tmp:
        base=Path(tmp)/'baseline'
        cmd(['git','worktree','add','--detach',str(base),BASE])
        try:
            for label,work in [('baseline',base),('proposed',ROOT)]:
                env=dict(os.environ); env['PYTHONPATH']=str(work/'src')
                xml=OUT/f'{label}.xml'
                p=cmd([sys.executable,'-m','pytest','-q','--junitxml',str(xml)],work,False,env)
                (OUT/f'{label}.log').write_text(p.stdout+p.stderr)
                if p.returncode not in (0,1) or not xml.exists():
                    raise RuntimeError(f'{label} suite did not run to a valid test conclusion, rc={p.returncode}')
                rows=[]
                for c in ET.parse(xml).getroot().iter('testcase'):
                    state='pass'
                    if c.find('failure') is not None:state='failure'
                    if c.find('error') is not None:state='error'
                    if c.find('skipped') is not None:state='skip'
                    rows.append([c.get('classname'),c.get('name'),state])
                results[label]={'returncode':p.returncode,'tests':sorted(rows),'count':len(rows)}
        finally:
            cmd(['git','worktree','remove','--force',str(base)],check=False)
    same=results['baseline']['tests']==results['proposed']['tests']
    failed=[r for r in results['proposed']['tests'] if r[2] in ('failure','error')]
    record={'scope':'ADMINISTRATIVE_NO_REGRESSION_NOT_SCIENTIFIC_QUALIFICATION',
       'baseline_commit':BASE,'proposed_commit':cmd(['git','rev-parse','HEAD']).stdout.strip(),
       'only_administrative_paths_changed':True,'changed_files':paths,
       'local_navigation_missing':missing,'administrative_tests_passed':True,
       'legacy_test_outcomes_identical':same,'legacy_failures_not_hidden':failed,
       'legacy_baseline_returncode':results['baseline']['returncode'],
       'legacy_proposed_returncode':results['proposed']['returncode'],
       'baseline_count':results['baseline']['count'],'proposed_count':results['proposed']['count'],
       'new_scientific_execution':False}
    (OUT/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    (OUT/'TEST_OUTCOMES.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if not same: raise RuntimeError('Housekeeping changed CPU outcomes; no merge admission')
if __name__=='__main__':main()
