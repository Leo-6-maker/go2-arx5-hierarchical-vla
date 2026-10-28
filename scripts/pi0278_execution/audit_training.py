"""Reuse the qualified journal audit; bind COVER93's box phase from its first micro."""
import argparse,importlib.util,inspect,json
from pathlib import Path
SOURCE=Path(__file__).with_name('qualified_training_audit.py')
spec=importlib.util.spec_from_file_location('qualified_training_audit',SOURCE);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
source=inspect.getsource(old.check_loss);assert source.count("'FOCUS_GRIP_BOUND_R3'")==1
exec(compile(source.replace("'FOCUS_GRIP_BOUND_R3'","'PI0278_COVER93_FINAL_PARENT_OBJECTIVE'"),str(SOURCE),'exec'),vars(old))
source=inspect.getsource(old.audit)
start=source.index("    phase_path=t/f'checkpoint-{steps}'/'FOCUS_GRIP_BOUND_R3.json'");end=source.index('    namespace=',start)
source=source[:start]+'''    box_rows=rows(t/'box_objective.jsonl')
    assert [b['forward'] for b in box_rows]==list(range(1,n+1))
    boxes={b['forward']:b for b in box_rows};phase=dict(phase='PI0278_COVER93_FINAL_PARENT_OBJECTIVE',resume_update=0)
'''+source[end:]
before="read(Path(read(root/'design/INPUT_BINDING.json')['old_root'])/'design/PROCESSED_INPUTS.json')['rows']"
assert source.count(before)==1;source=source.replace(before,"read(root/'design/ELIGIBLE_MEMBERSHIP.json')['rows']")
source=source.replace("'_PLUS_POINT1_BOX_FROM_UPDATE1803'","'_PLUS_POINT1_BOX_FROM_FIRST_COVER93_MICROBATCH'")
exec(compile(source,str(SOURCE),'exec'),vars(old))

def check():
    box=dict(objective_phase='PI0278_COVER93_FINAL_PARENT_OBJECTIVE',additional_generation=False,box=.4,weighted_box=.04,total=2.34)
    old.check_loss({'total':2},{'total':2.34,'weighted':.3},box)
    for bad in (dict(box,weighted_box=.08),dict(box,total=2.38),dict(box,additional_generation=True)):
        try:old.check_loss({'total':2},{'total':2.34,'weighted':.3},bad)
        except AssertionError:pass
        else:raise AssertionError('Wrong coefficient/total/extra generation admitted')
    print('PASS_FINAL_PARENT_OBJECTIVE_IDENTITY_AND_NEGATIVE_CHECKS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    else:assert a.root;old.audit(a.root,'COVER93')
