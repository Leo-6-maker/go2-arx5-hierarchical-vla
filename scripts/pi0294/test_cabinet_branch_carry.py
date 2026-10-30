"""Actual native closed book pool, preserving all prior charges and STOP."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.multitask import cabinet_branch as b,night_scope as s,source_stage_bundle as bundle


@pytest.fixture(scope='module')
def actual():
    d=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not d:pytest.skip('Exact closed source native capture required')
    root=Path(d);v=json.loads((root/'CABINET_UNRUN_BRANCH_CLOSED_POOL_ACTUAL.private.json').read_bytes())
    names=dict(account='control/NIGHT_SOURCE_ACCOUNT.json',lease='control/NATIVE_SOURCE_LEASE.json',
        terminal='outputs/SOURCE_CHAIN_TERMINAL.json',stop='STOP.json')
    bodies={k:base64.b64decode(v['records'][n]['base64'])for k,n in names.items()}
    auth=(root.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    destination='/var/cache/source/20261030-cabinet-unrun-night-source-four01'
    return bodies,auth,destination


def test_actual_one_native_pool_carries_all_unknown_upper_and_only_two_cabinet_tasks(actual):
    bodies,auth,destination=actual;v=b.account(bodies,auth,destination=destination,now=s.SUCCESSOR_STOP-3600)
    old=json.loads(bodies['account']);assert v['charged_upper']==old['charged_upper']
    assert v['operations']==old['operations']and v['caps']==old['caps']and not v['native_adopted']
    closed=b.transfer(bodies['account'],(json.dumps(v,indent=2)+'\n').encode(),destination=destination)
    assert closed['custody']=='CLOSED_CARRIED_TO_ONE_UNRUN_CABINET_BRANCH'
    assert closed['charged_upper']==old['charged_upper']and closed['operations']==old['operations']
    assert bundle.plan('ROBOCASA_CABINET')==[dict(task_kind='ROBOCASA_CABINET',side=x)for x in('right','left')]


@pytest.mark.parametrize('fault',['refund','remove_old_operation','book_retry','different_STOP','clock_extension'])
def test_no_replenished_pool_book_retry_or_safety_evidence_substitution(actual,fault):
    bodies,auth,destination=actual
    if fault=='different_STOP':
        bad=dict(bodies,stop=b'{}')
        with pytest.raises(PermissionError):b.account(bad,auth,destination=destination,now=s.SUCCESSOR_STOP-1)
        return
    v=b.account(bodies,auth,destination=destination,now=s.SUCCESSOR_STOP-1)
    if fault=='refund':v['charged_upper']['physics_integrations']-=1
    elif fault=='remove_old_operation':v['operations']=v['operations'][1:]
    elif fault=='book_retry':v['operations'].append(dict(kind='ATTEMPT',task='LIBERO_BOOK_RIGHT',operation='LIBERO_RIGHT_FIRST01'))
    else:v['deadline_epoch']+=1
    with pytest.raises(PermissionError):s.validate_account(v)
