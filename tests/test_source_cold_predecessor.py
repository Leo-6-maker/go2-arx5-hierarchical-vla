import base64,json
from types import SimpleNamespace
import pytest
from scripts.pi0294.multitask.night_source_stage import PROGRAM


def check(root,*,cold,state,results=None):
    code=' facts=dict'+PROGRAM.split(' facts=dict',1)[1].split(" for name,pin in P.get('predecessor_preserved_files'",1)[0]
    code='if True:\n'+code
    terminal=dict(results=[]if results is None else results,cold_never_started=cold)
    return exec(compile(code,'actual_cold_predecessor_fence','exec'),dict(
        q=SimpleNamespace(returncode=0,stdout='MainPID=0\nSubState='+state+'\n'),
        previous=root,carry={'terminal_base64':base64.b64encode(json.dumps(terminal).encode()).decode()},
        base64=base64,json=json))


def test_actual_cold_predecessor_requires_no_native_intent_or_started_record(tmp_path):
    check(tmp_path,cold=True,state='dead')
    check(tmp_path,cold=False,state='exited',results=[{'actual_run':True}])
    (tmp_path/'control').mkdir()
    for name in ('NATIVE_LAUNCH_INTENT.json','ACTUAL_DETACHED_OWNER_STARTED.json'):
        p=tmp_path/'control'/name;p.write_text('{}')
        with pytest.raises(AssertionError):check(tmp_path,cold=True,state='dead')
        p.unlink()


@pytest.mark.parametrize('cold,state,results',[(False,'dead',[]),(True,'exited',[]),(True,'dead',[{'actual_run':True}])])
def test_completed_run_cannot_be_relabelled_as_never_started(tmp_path,cold,state,results):
    with pytest.raises(AssertionError):check(tmp_path,cold=cold,state=state,results=results)
