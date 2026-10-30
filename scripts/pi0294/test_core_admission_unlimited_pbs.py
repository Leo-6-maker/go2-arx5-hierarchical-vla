"""Replay the real bound05 inputs; historical CPU replay grants no admission."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_admission as admission,mix_core_accounting as c,mix_accounting as a


@pytest.fixture(scope='module')
def actual_inputs():
    directory=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('actual bound05 captures required')
    root=Path(directory)/'MAINTENANCE30_CONTROL_IO'
    read=lambda n:json.loads((root/(n+'.private.json')).read_bytes())
    p=read('NODE_LOCAL05_RUNTIME_ASSEMBLY_FINAL_PREPARED');b=read('NL05_ACTUAL_BIND_PROPOSAL');cap=read('NL05_PRE_GRANT_TERMINAL_CAPTURE')
    g=json.loads(gzip.decompress(base64.b64decode(p['proposed_GLOBAL_gzip_base64'])))
    l=json.loads(gzip.decompress(base64.b64decode(p['proposed_LOCAL_gzip_base64'])))
    for value in(g,l):value[c.FIELD][b['row']['request_id']]=deepcopy(b['row'])
    g['revision']+=1
    assert a.sha(a.raw(g))==b['GLOBAL_after_sha256']and a.sha(a.raw(l))==b['LOCAL_after_sha256']
    return p,b,cap,g,l,read('NL05_BIND_GLOBAL_PROOF')


def test_real_bound_native_replay_preserves_all14_and_handles_human_unlimited_count(actual_inputs):
    from scripts.pi0294.pv1.evaluation_admission import resource_floor
    p,b,cap,g,l,gproof=actual_inputs;row=p['assembly']['rows']['hopper']
    used,held=resource_floor(g,l);cfg=p['owner']['config']
    used['Core_storage_bytes']=max(used['Core_storage_bytes'],cfg['Core_storage_upper_bytes'])
    used['local_storage_bytes']=max(used['local_storage_bytes'],cfg['LOCAL_storage_upper_bytes'])
    now=gproof['observed_epoch']+10
    snapshot=dict(schema=admission.SNAPSHOT,task=cfg['marker']['task'],scope_grant_sha256=p['owner']['manifest']['authority']['scope_grant_sha256'],T_CD=cfg['marker']['T_CD'],deadline_epoch=cfg['marker']['deadline_epoch'],GLOBAL_sha256=a.sha(a.raw(g)),LOCAL_sha256=a.sha(a.raw(l)),observed_epoch=gproof['observed_epoch'],heartbeat_only_raw_ledger=False,used=used,reserved=held,Core_storage_upper_bytes=used['Core_storage_bytes'],LOCAL_storage_upper_bytes=used['local_storage_bytes'],actual_current_Core_pending={},historical_storage_witness_sha256=cfg['storage_witness_sha256'],historical_CPU_replay_not_runtime=True)
    files={n:base64.b64decode(v)for n,v in row['files'].items()}
    result=admission.recompute(manifest_bytes=base64.b64decode(row['manifest_bytes_base64']),request_bytes=a.raw(p['assembly']['request']),global_bytes=a.raw(g),local_bytes=a.raw(l),snapshot_bytes=a.raw(snapshot),training_capture_bytes=files['plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json'],population_bytes=files['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'],plan_bytes=files['plans/SOURCE_EVALUATION_PLAN.json'],prior_bytes=files['plans/PRIOR_EVALUATION_OUTCOMES.json'],native_facts_bytes=base64.b64decode(cap['prebind_process_base64']),now_epoch=now)
    assert result['shortfalls']=={}and len(result['dimensions'])==14
    pbs=result['dimensions']['PBS_submissions'];assert pbs['cap']is None and pbs['shortfall']==0 and pbs['count_still_accounted']
    forecast=result['recovered_full_campaign_future']['dimensions']
    assert all(result['dimensions'][k]['cap']==v['cap']for k,v in forecast.items())
    assert not result['model_permission']and not result['physics_authorized']and not result['native_or_scientific_admission']


def test_only_pbs_can_have_no_cap_and_actual_floors_still_increase_projection(actual_inputs):
    _,_,_,g,_,_=actual_inputs
    full=dict(projected=100,used=10,reserved=20,cap=105)
    value=admission.adjusted_full_dimension(g,'Core_CPU_s',full,17,24)
    assert value['projected']==111 and value['shortfall']==6 and value['cap']==105
    unlimited=dict(full,cap=None,PBS_submission_count_limit=None,count_still_accounted=True)
    with pytest.raises(PermissionError):admission.adjusted_full_dimension(g,'Core_CPU_s',unlimited,17,24)
    without=deepcopy(g);without.pop('V6_human_GPU_PBS_count_unlimited')
    with pytest.raises(PermissionError):admission.adjusted_full_dimension(without,'PBS_submissions',unlimited,17,24)
