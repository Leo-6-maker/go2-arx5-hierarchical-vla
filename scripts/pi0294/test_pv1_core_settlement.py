"""Known failed native effects preserve actual costs and original history."""
from copy import deepcopy
import json

import pytest

from scripts.pi0294.pv1 import evaluation_core_settlement as s,evaluation_core_accounting as c
from scripts.pi0294.pv1 import evaluation_admission as a
from scripts.pi0294.pv1 import evaluation_core_admission as admission
from scripts.pi0294.test_pv1_core_evaluation_admission import captured_inputs
from scripts.pi0294.test_pv1_core_evaluation_accounting import process,final,NOW


def inputs(captured):
    args,record=captured;proposal=admission.reserve_pair_proposal(inputs=args,record=record)
    g=c.bind(proposal['proposed_global'],request_id=record['request_id'],process=process(record),now_epoch=NOW)
    native,effects=final(g)
    transfer=dict(schema='PV1_CORE_MODEL_TRANSFER_TERMINAL_READBACK_V1',request_id=record['request_id'],
        request_sha256=record['request_sha256'],source_sha=record['source_sha'],all_effects_accounted=True,
        unknown_effects=[],accounted_bytes=2*c.MODEL_BYTES)
    return dict(global_bytes=a.canonical(g),local_bytes=a.canonical(proposal['proposed_LOCAL']),
        request_id=record['request_id'],native_proof=native,effects=effects,transfer_proof=transfer,now_epoch=NOW+10)


def test_once_actual_known_failed_effects_in_both_original_ledgers(captured_inputs):
    values=inputs(captured_inputs);before=deepcopy(values);oldg=json.loads(values['global_bytes']);oldl=json.loads(values['local_bytes'])
    result=s.propose(**values);g,l=result['proposed_global'],result['proposed_LOCAL']
    assert values==before and result['IO_performed'] is False
    assert g['PBS_submissions']==oldg['PBS_submissions'] and l['PBS_submissions']==oldl['PBS_submissions']
    assert g['training_runs']==oldg['training_runs'] and g['optimizer_updates']==11714
    assert g['used']['H200_s']==oldg['used']['H200_s']+70
    assert l['GPU_allocated_s']==oldl['GPU_allocated_s']+70
    assert l['CPU_s']['CoreLab']==oldl['CPU_s']['CoreLab']+20
    assert l['CPU_s']['Hopper']==oldl['CPU_s']['Hopper']+20
    assert l['neural_queries']==oldl['neural_queries']+3
    assert l['network_transfer_accounted_bytes']==oldl['network_transfer_accounted_bytes']+2*c.MODEL_BYTES
    assert l['Core_native_GPU_current_storage_bytes']==1024
    assert l['GPU_reserved_s']==0 and all(v==0 for v in l['CPU_reserved_s'].values())
    assert l['Core_native_GPU_reserved']==dict(CoreLab_output_bytes=0,transfer_bytes=0)
    with pytest.raises(PermissionError):s.propose(**dict(values,global_bytes=a.canonical(g),local_bytes=a.canonical(l)))


@pytest.mark.parametrize('fault',['unknown_transfer','wrong_transfer_request','excess_transfer','missing_GPU_hold',
    'missing_Core_CPU_hold','missing_output_hold','missing_transfer_hold','already_settled','timeout_not_exit'])
def test_unknown_or_missing_hold_never_permits_refund_or_duplicate_settlement(captured_inputs,fault):
    v=inputs(captured_inputs);l=json.loads(v['local_bytes']);t=v['transfer_proof']
    if fault=='unknown_transfer':t['unknown_effects']=['HF_UNKNOWN']
    if fault=='wrong_transfer_request':t['request_sha256']='f'*64
    if fault=='excess_transfer':t['accounted_bytes']+=1
    if fault=='missing_GPU_hold':l['GPU_reserved_s']=0
    if fault=='missing_Core_CPU_hold':l['CPU_reserved_s']['CoreLab']=0
    if fault=='missing_output_hold':l['Core_native_GPU_reserved']['CoreLab_output_bytes']=0
    if fault=='missing_transfer_hold':l['Core_native_GPU_reserved']['transfer_bytes']=0
    if fault=='already_settled':l['Core_native_GPU_reservations'][v['request_id']]['status']='SETTLED_KNOWN_NATIVE_TERMINAL'
    if fault=='timeout_not_exit':v['native_proof']['exit_evidence_kind']='POLL_TIMEOUT'
    v['local_bytes']=a.canonical(l)
    with pytest.raises(PermissionError):s.propose(**v)


def test_unknown_generation_is_paid_and_stays_unknown_not_clean_completion(captured_inputs):
    v=inputs(captured_inputs);v['effects']['unknown_query_intents']=[1]
    out=s.propose(**v);g,l=out['proposed_global'],out['proposed_LOCAL']
    assert g['reserved']['neural_queries']==1 and g['unknown_effects']
    assert l['unknown_effects'][-1]['indices']==[1]
    assert out['scientific_or_native_admission'] is False
