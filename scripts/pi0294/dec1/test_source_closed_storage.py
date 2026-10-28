"""Closed financial/known storage projection on synthetic byte-bound receipts."""
from copy import deepcopy
import pytest
from scripts.pi0294.dec1 import source_closed_storage as unit
from scripts.pi0294.dec1.test_source_closed_costs import facts


def raw(f):
    return dict(parent_bytes=unit.costs.encode(f['p']),Core_bytes=unit.costs.encode(f['c']),
        Hopper_bytes=unit.costs.encode(f['d']),closures=f['closures'])


def packet(f):
    costs=unit.costs
    first=costs.propose(**raw(f))
    f['d'].clear();f['d'].update(first['proposed_Hopper_DEC1'])
    for ledger in (f['p'],f['c'],f['d']):
        if ledger is f['p']:ledger['T0_epoch']=1792348196
        else:ledger['original_T0_epoch']=1792348196
    data=raw(f)
    s=dict(authority_release=costs.AUTHORITY,campaign_run_uuid='campaign',T0_epoch=1792348196,
        actual_DEC1_ledger_sha256=costs.sha(data['Hopper_bytes']),
        used=dict(Core_storage_bytes=100*1024**3,H200_s=77777),
        dec1_used=dict(Core_storage_bytes=80*1024**3,local_storage_bytes=8*1024**3,optimizer_updates=7939),
        storage_peaks=dict(Core_storage_bytes=100*1024**3),dec1_storage_peaks=dict(Core_storage_bytes=80*1024**3),
        frozen_training_state=dict(A=4096,B=2048,old_failed_updates=771))
    return s,dict(data,allocated_local_namespace_bytes=2*1024**3,observed_epoch=100,now_epoch=101)


def test_both_closed_kinds_added_once_without_training_mutation(facts):
    s,args=packet(facts);before=deepcopy(s);out=unit.augment(s,**args)
    assert out['used']['Core_storage_bytes']==100*1024**3+1000
    assert out['dec1_used']['Core_storage_bytes']==80*1024**3+1000
    assert out['closed_source_storage_projection']['total_closed_Core_output_bound_added_once']==1000
    assert out['frozen_training_state']==s['frozen_training_state'] and s==before
    assert out['dec1_used']['optimizer_updates']==7939 and out['used']['H200_s']==77777
    with pytest.raises(PermissionError,match='ONE_APPLICATION'):unit.augment(out,**args)


def test_missing_real_catchup_cannot_be_replaced_with_storage_only(facts):
    s,args=packet(facts);facts['d'][unit.costs.NEW_MARKER]={}
    data=raw(facts);args.update(data);s['actual_DEC1_ledger_sha256']=unit.costs.sha(data['Hopper_bytes'])
    with pytest.raises(PermissionError,match='COST_CATCHUP'):unit.augment(s,**args)


def test_active_work_cannot_be_silently_projected_as_closed_zero_pending(facts):
    s,args=packet(facts);u=unit.costs
    for ledger in (facts['p'],facts['c']):
        row=ledger['source_cpu_reservations'][u.NEW]
        row.update(status='RUNNING',remaining=deepcopy(row['requested']))
        row.pop('known_terminal_settlement_sha256');row.pop('scientific_verdict')
        ledger['source_cpu_reserved']=deepcopy(row['requested']);del ledger[u.KNOWN_INDEX][u.NEW]
    del facts['closures'][u.NEW];del facts['d'][u.NEW_MARKER][u.NEW]
    data=raw(facts);args.update(data);s['actual_DEC1_ledger_sha256']=u.sha(data['Hopper_bytes'])
    with pytest.raises(PermissionError,match='ACTIVE_SOURCE_NEEDS_COMPLETE_SEPARATE_NATIVE_PENDING_ADAPTER'):
        unit.augment(s,**args)


@pytest.mark.parametrize('fault',['already_old_augmented','namespace_uncovered','stale','wrong_Hopper'])
def test_no_reset_hidden_outputs_or_stale_byte_evidence(facts,fault):
    s,args=packet(facts)
    if fault=='already_old_augmented':s['used']['Core_storage_bytes']+=500
    if fault=='namespace_uncovered':args['allocated_local_namespace_bytes']=8*1024**3
    if fault=='stale':args['observed_epoch']=0;args['now_epoch']=1800
    if fault=='wrong_Hopper':s['actual_DEC1_ledger_sha256']='f'*64
    with pytest.raises(PermissionError):unit.augment(s,**args)
