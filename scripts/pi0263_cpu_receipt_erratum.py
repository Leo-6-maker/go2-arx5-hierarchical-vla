"""Preserve and reconcile a completed CPU receipt's scope; no model/metric replay."""
import ast
import collections
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0263')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_scope_code():
    # Execute the actual receipt expressions after a representative loop overwrites point.
    tree=ast.parse((ROOT/'code/pi0263_cpu.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
              and n.func.id=='dict' and any(k.arg=='reused_donor_rows' for k in n.keywords))
    candidate=next(k.value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=='dict' for k in n.keywords if k.arg=='synthetic_research_candidate')
    for donor,requested,points,status,reused in (
        (False,None,['P2000','F2000','X2000'],'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY',1152),
        (True,None,['P2000'],'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_DONOR_1152_REPLAY',1152),
        (False,'F2000',['F2000'],'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_F2000_1152_REPLAY',0),
        (False,'X2000',['X2000'],'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_X2000_1152_REPLAY',0)):
        env=dict(donor_only=donor,requested_point=requested,point=points[-1],points=points,
                 qualification=[dict(point=p,combined_pass=True) for p in points])
        for key,expected in (('status',status),('reused_donor_rows',reused)):
            value=next(k.value for k in call.keywords if k.arg==key)
            assert eval(compile(ast.Expression(value),'<actual-scope-expression>','eval'),env)==expected
        assert eval(compile(ast.Expression(candidate),'<actual-candidate-expression>','eval'),env)==('F2000' if len(points)==3 else 'NONE')


def main():
    check_scope_code()
    path=ROOT/'CPU_REPLAY.json'
    preserved=ROOT/'evidence/cpu_schema_erratum'
    preserved.mkdir(parents=True,exist_ok=True)
    raw=preserved/'RAW_CPU_REPLAY.json'
    if not raw.exists():
        shutil.copy2(path,raw)
    receipt=json.loads(raw.read_text())
    assert receipt['status']=='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_X2000_1152_REPLAY' and receipt['rows']==3456
    original_code=preserved/'pi0263_cpu_as_replayed.py'
    assert sha(original_code)==receipt['code_sha256']
    assert receipt['eval_card_sha256']==sha(ROOT/'UNION_EVAL_CARD.json')
    assert len(receipt['provenance'])==3456
    seen=set()
    for row in receipt['provenance']:
        key=row['point'],row['uid'];assert key not in seen;seen.add(key)
        cell=ROOT/'eval'/row['point']/'cells'/row['uid']
        assert sha(cell/'record.json')==row['record_sha256'] and sha(cell/'tensors.npz')==row['tensor_sha256']
    assert seen=={(p,f'u-{i:04d}') for p in ('P2000','F2000','X2000') for i in range(1152)}
    for name,n in (('LEGACY_PANEL_ROWS.csv',680),('CROSSED_PANEL_ROWS.csv',640)):
        with (ROOT/name).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert collections.Counter(r['point'] for r in rows)==dict.fromkeys(('P2000','F2000','X2000'),n)
    qualification=json.loads((ROOT/'QUALIFICATION.json').read_text())
    assert not any(r['combined_pass'] for r in qualification['points']) and qualification['synthetic_research_candidate']=='NONE'
    assert receipt['reused_donor_rows']==0 and receipt['new_attempts']==2232 and receipt['heads']==8928
    corrected={**receipt,'status':'PASS_INDEPENDENT_COMPUTATION_PATH_CPU_3456_REPLAY','reused_donor_rows':1152,
        'scope_erratum':dict(raw_receipt_sha256=sha(raw),raw_receipt='evidence/cpu_schema_erratum/RAW_CPU_REPLAY.json',
            reason='Loop cursor shadowed requested point after all3456 rows passed; scope/status and donor reuse were mislabeled.',
            original_computation_code_sha256=sha(original_code),fixed_scope_code_sha256=sha(ROOT/'code/pi0263_cpu.py'),
            reconciliation_code_sha256=sha(Path(__file__)),checked_saved_records=3456,model_queries=0,
            metric_recomputations=0,quality_changed=False,actual_expression_scope_checks=4)}
    path.write_text(json.dumps(corrected,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS_PRESERVED_RAW_CPU_SCOPE_ERRATUM',rows=3456,model_queries=0,quality_changed=False)))


if __name__=='__main__':
    main()
