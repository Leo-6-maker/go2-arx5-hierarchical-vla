"""Phase marker parsing: completed24/lease fields here are explicit CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_pilot_scope as s,minimal_scope as m
from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a,successor_cd
from scripts.pi0294.pv1.dataset import POPULATION

@pytest.fixture
def actual():
    path=os.environ.get('ARX5_CORE_RPC_LIMIT_CAPTURE')
    if not path:pytest.skip('captured actual closed C10 GLOBAL and LOCAL required')
    p=json.loads(Path(path).read_bytes())['value'];return json.loads(gzip.decompress(base64.b64decode(p['GLOBAL_gzip_base64'])))

def completed_fixture(g):
    g=deepcopy(g);p=g[a.PACKAGE];p['completed_stages'].append(c.key('C50'));rid='ed3880cd-2a21-4855-84aa-ec8d943104cb';row=g[c.FIELD][rid];row['native_exit_status']=0
    history=dict(schema='V5_CD_ORIGINAL_DEV24_VALIDATED_HISTORY_V1',authority_release=successor_cd.TASK_SOURCE,candidate='C50',population_plan_sha256=POPULATION,checkpoint_manifest_sha256=row['checkpoint_manifest_sha256'],checkpoint_step=1024,complete_DEV24=True,valid_conditions=24,remaining_eligible_conditions=0,condition_result_sha256={str(i):'c'*64 for i in range(24)},independent_audit_sha256={str(i):'d'*64 for i in range(24)})
    p['closed_Eval_stages']={c.key('C50'):dict(schema='V5_CD_COMPLETE_ORIGINAL_DEV24_AUDITED_DUAL_CLOSED_V1',request_id=rid,candidate='C50',terminal_native_sha256=row['terminal_native_sha256'],checkpoint_manifest_sha256=row['checkpoint_manifest_sha256'],all_valid_failures_preserved=True,history=history,GPU_lease_sha256='a'*64,physical_lease_sha256='b'*64)}
    return g

def test_actual_engineering_terminal_still_requires_both_original_panels(actual):
    assert s.original_remaining_panels(actual)==['C50','D25']

def test_explicit_closed24_fixture_keeps_D_and_needs_no_success_threshold(actual):
    g=completed_fixture(actual);h=g[a.PACKAGE]['closed_Eval_stages'][c.key('C50')]['history'];h['successes']=dict(T1=0,T2=0)
    assert s.original_remaining_panels(g)==['D25']

@pytest.mark.parametrize('kind',['stage-only','partial','wrong-checkpoint','unknown','failed','lease','unvalidated-history'])
def test_no_credit_for_engineering_partial_unknown_or_only_completed_string(actual,kind):
    g=completed_fixture(actual);p=g[a.PACKAGE];proof=p['closed_Eval_stages'][c.key('C50')];row=g[c.FIELD][proof['request_id']]
    if kind=='stage-only':p.pop('closed_Eval_stages')
    elif kind=='partial':proof['history']['valid_conditions']=23
    elif kind=='wrong-checkpoint':proof['checkpoint_manifest_sha256']='f'*64
    elif kind=='unknown':row['unknown_query_intents']=[0]
    elif kind=='failed':row['native_exit_status']=1
    elif kind=='lease':proof['GPU_lease_sha256']=None
    else:proof['history']['schema']='SUMMARY_NOT_VALIDATED_ORIGINAL24'
    with pytest.raises(PermissionError):s.original_remaining_panels(g)
