"""Synthetic authority corruption tests; no native task evidence."""
from datetime import datetime
import json
from pathlib import Path

import pytest

from . import task_path_recovery as recovery
from .task_path_runtime import cards, cards_for_manifest, require_card

ROOT = '/node/runs/PI0294/20261029T160000Z-standing-task-path-s1'
REPORTS = Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022'


def fixtures(monkeypatch):
    raw = (REPORTS/'GO2_ARX5_S1_TWO_TASK_RECOVERY_HUMAN_APPROVAL_20261029.json').read_bytes()
    original = (REPORTS/'GO2_ARX5_S1_HUMAN_APPROVAL_20261029.json').read_bytes()
    account = json.dumps(dict(scope_status='CLOSED_ENGINEERING_INCOMPLETE_NO_QUALIFIED_SHORT',
        native_attempts_admitted=False, starts=3, integrations=5006, RGB_pairs=0)).encode()
    lease = json.dumps(dict(status='CLOSED', run_uuid=recovery.OLD_RUN)).encode()
    monkeypatch.setattr(recovery, 'CLOSED_ACCOUNT_PIN', recovery.digest(account))
    monkeypatch.setattr(recovery, 'CLOSED_LEASE_PIN', recovery.digest(lease))
    now = datetime.fromisoformat(json.loads(raw)['adoption_recorded_at_UTC']).timestamp()
    return raw, original, account, lease, now


def test_exact_new_approval_keeps_old_clock_and_all_caps(monkeypatch):
    raw, original, account, lease, now = fixtures(monkeypatch)
    value = recovery.recovery_authority(raw, original, account, lease, run_root=ROOT, now_epoch=now)
    assert recovery.digest(raw) == recovery.APPROVAL_PIN
    assert value['approved'] == recovery.CAPS
    assert 5006 + 2*(18500+3) == recovery.LIMITS['integrations']
    assert 2*641 == recovery.LIMITS['RGB_pairs']
    assert value['window_end_UTC'] == json.loads(original)['window_end_UTC']


@pytest.mark.parametrize('field,value', [('window_end_UTC','2026-11-07T00:00:00+00:00'),
    ('answer','同意任意重试'), ('old_closed_scope_reopened',True), ('scope','OTHER')])
def test_changed_approval_cannot_authorize_scope(monkeypatch,field,value):
    raw, original, account, lease, now = fixtures(monkeypatch)
    changed=json.loads(raw);changed[field]=value
    with pytest.raises(PermissionError, match='EXACT_NEW_HUMAN_APPROVAL'):
        recovery.recovery_authority(json.dumps(changed).encode(),original,account,lease,run_root=ROOT,now_epoch=now)


@pytest.mark.parametrize('which', ['old_account','old_lease','original'])
def test_old_proofs_and_original_authority_must_be_exact(monkeypatch,which):
    raw, original, account, lease, now=fixtures(monkeypatch)
    if which=='old_account':account+=b' '
    elif which=='old_lease':lease+=b' '
    else:original+=b' '
    with pytest.raises(PermissionError):
        recovery.recovery_authority(raw,original,account,lease,run_root=ROOT,now_epoch=now)


def test_old_run_cannot_be_reopened_and_no_expiry_extension(monkeypatch):
    raw, original, account, lease, now=fixtures(monkeypatch)
    for root,at in [(str(Path(ROOT).parent/recovery.OLD_RUN),now),(ROOT,now-.001),
        (ROOT,datetime.fromisoformat(json.loads(raw)['window_end_UTC']).timestamp())]:
        with pytest.raises(PermissionError):
            recovery.recovery_authority(raw,original,account,lease,run_root=root,now_epoch=at)


def test_only_two_exact_recovery_cards_and_no_old_confirmation(tmp_path):
    manifest=dict(recovery_scope=recovery.SCOPE,recovery_approval_sha256=recovery.APPROVAL_PIN,
        code=dict(vla_commit='a'*40))
    expected=recovery.recovery_cards()
    assert cards_for_manifest({})==cards()
    assert cards_for_manifest(manifest)==expected
    assert [c['id']for c in expected]==['S1_RECOVERY01_T1','S1_RECOVERY01_T2']
    assert all(c['duration_s']==32 and c['warmup_sim_s']==5 and c['base_translation_preference_m']==0 for c in expected)
    folder=tmp_path/'native-s1';folder.mkdir()
    (folder/'FIXED_SERIES_CARDS.json').write_text(json.dumps(dict(cards=expected,source_sha='a'*40)))
    assert require_card(tmp_path,manifest,'S1_RECOVERY01_T1')==expected[0]
    for forbidden in ('S1_SHORT_T1','S1_CONFIRM_T2','S1_RECOVERY02_T1'):
        with pytest.raises(PermissionError):require_card(tmp_path,manifest,forbidden)
    with pytest.raises(PermissionError):cards_for_manifest(dict(manifest,recovery_scope='OTHER'))


def test_carry_forward_cannot_drop_old_costs_or_attempts(tmp_path,monkeypatch):
    old=dict(starts=3,integrations=5006,RGB_pairs=0,Core_CPU_upper_charged_s=834,
        PC_CPU_upper_charged_s=500,transfer_upper_charged_bytes=202663168,
        Core_staging_upper_charged_bytes=236814886,PC_evidence_upper_charged_bytes=536870912,
        native_attempts=dict(S1_SHORT_T2=dict(status='NATIVE_TERMINAL_ONCE_SETTLED')))
    current=dict(old,recovery_scope=recovery.SCOPE,carry_forward_account_sha256=recovery.CLOSED_ACCOUNT_PIN)
    (tmp_path/'control').mkdir()
    for name in ('S1_RECOVERY_HUMAN_APPROVAL.json','S1_HUMAN_APPROVAL.json','CLOSED_S1_NATIVE_LEASE.json'):
        (tmp_path/name).write_bytes(b'bound-by-authority')
    (tmp_path/'CLOSED_S1_ACCOUNT_CARRY_FORWARD.json').write_text(json.dumps(old))
    monkeypatch.setattr(recovery,'recovery_authority',lambda *args,**kwargs:None)
    manifest=dict(recovery_scope=recovery.SCOPE,recovery_approval_sha256=recovery.APPROVAL_PIN,
        carry_forward_account_sha256=recovery.CLOSED_ACCOUNT_PIN)
    path=tmp_path/'control/S1_PREPARATION_ACCOUNT.json';path.write_text(json.dumps(current))
    assert recovery.validate_recovery(tmp_path,manifest,0)==recovery.LIMITS
    for corrupted in (dict(current,starts=0),dict(current,PC_CPU_upper_charged_s=0),dict(current,native_attempts={})):
        path.write_text(json.dumps(corrupted))
        with pytest.raises(PermissionError,match='COSTS_RETAINED'):recovery.validate_recovery(tmp_path,manifest,0)
