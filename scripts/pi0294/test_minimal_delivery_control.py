"""Captured actual history with explicitly fictional future512/OS0 artifacts.

No future native model, F, lease close, source role, HF commit or adoption is
claimed. The real complete original validators/14-dimension forecast run here.
"""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_delivery_control as d,minimal_inference_transfer as transfer
from scripts.pi0294.generalization import minimal_inference_export as export,minimal_execution as e
from scripts.pi0294.generalization import minimal_remaining_execution as r,minimal_remaining_terminal as terminal
from scripts.pi0294.test_minimal_remaining_execution import terminal_packet


@pytest.fixture(scope='module')
def captured():
    directory=os.environ.get('ARX5_CURRENT_REMAINING_CAPTURE_DIR')
    if not directory:pytest.skip('exact current remaining GLOBAL/private historical LOCAL capture required')
    m=Path(directory);p=sorted(m.glob('UNSUBMITTED_RESUME01_CURRENT_CANONICAL_EXACT_RECONSTRUCTED_*.private.json'))[-1]
    cap=json.loads(p.read_bytes());gb=gzip.decompress(base64.b64decode(cap['GLOBAL_gzip_base64']));assert e.sha(gb)==cap['GLOBAL_sha256']
    local=json.loads((m/'D25_SAME64_WIRE01_PREPARED.private.json').read_bytes());lb=gzip.decompress(base64.b64decode(local['LOCAL_gzip_base64']))
    prior=json.loads((m.parent/'v5-execution-20261025/C50_REAL_CLOSED1024_INFERENCE_TRANSFER_PREPARED.private.json').read_bytes())
    readback=json.loads(base64.b64decode(prior['private_restoration']['closed_metadata_readback_base64']))
    small={n:base64.b64decode(readback['files'][n]['base64'])for n in transfer.original.SMALL}
    assert e.sha(small['statistics.json'])==e.R0
    return gb,lb,small


@pytest.fixture(scope='module')
def future(captured):
    gb,lb,old_small=captured;g=json.loads(gb);index,row=r.basic(g);native,effects,audit=terminal_packet((g,None,row))
    when=1793235000.;native.update(finished_epoch=when-3,observed_epoch=when-2,
        conservative_GPU_s=int(when-3-row['submitted_epoch'])+120,
        conservative_CPU_s=10000,new_native_conservative_CPU_s=10000)
    exports={};witness=[]
    for i,arm in enumerate(e.training.ARMS):
        endpoint=audit['endpoints'][arm];endpoint['checkpoint_manifest_sha256']=str(i+1)*64
        reference=dict(schema=export.SCHEMA,arm=arm,global_step=512,cursor=16384,scheduler_horizon=1024,
            statistics_sha256=e.R0,raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,source_sha=row['source_sha'],
            wrapped_checkpoint_manifest_sha256=endpoint['checkpoint_manifest_sha256'],full_resume_checkpoint_retained=True,
            ordinary_upstream_full_model=True,adapter_only_file=False)
        shard='model-00001-of-00001.safetensors';files=dict(old_small)
        files.update({'model.safetensors.index.json':e.raw({'weight_map':{'fixture_weight':shard}}),
            'FULL_SOURCE_CHECKPOINT_REFERENCE.json':e.raw(reference),shard:b'OPAQUE_FUTURE512_CPU_FIXTURE_ONLY_'+arm.encode()})
        complete=dict(schema=export.SCHEMA,arm=arm,global_step=512,cursor=16384,scheduler_horizon=1024,
            full_resume=False,original_full_resume_retained=True,statistics_sha256=e.R0,
            raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
            wrapped_checkpoint_manifest_sha256=endpoint['checkpoint_manifest_sha256'],files={n:e.sha(b)for n,b in files.items()})
        files['CHECKPOINT_COMPLETE.json']=e.raw(complete);endpoint['official_inference_manifest_sha256']=e.sha(files['CHECKPOINT_COMPLETE.json'])
        root='/fixture/'+row['run_uuid']+'/inference/'+arm
        witness.extend(dict(path=root+'/'+n,bytes=len(b),sha256=e.sha(b))for n,b in files.items())
        exports[arm]=dict(root=root,files=files)
    audit['streamed_witnesses']=dict(paths=witness,all_paths_rechecked=True,all_unique_inodes_independently_rehashed=True)
    physical=deepcopy(index['physical_preflight']);physical['observed_epoch']=when-1
    physical['new_allocated_bytes']=min(80*1024**3,physical['paid_output_ceiling']-physical['prefix_allocated_bytes']-physical['next_output_upper_bytes'])
    g=terminal.settle(g,row['request_id'],native=native,effects=effects,audit=audit,
        physical_proof=physical,physical_observed_epoch=when-1,now_epoch=when-1)
    settled=e.raw(dict(native=native,effects=effects,audit=audit))
    lease=e.raw(dict(status='CLOSED',run_uuid=row['run_uuid'],manifest_sha256=row['run_manifest_sha256'],
        terminal_receipt_sha256=e.sha(settled),closed_epoch=when))
    observation=dict(epoch=when+2,GLOBAL_sha256=e.sha(e.raw(g)),same_owner_live=False,same_wrapper_live=False,
        effects={k:row[k]for k in('request_id','request_sha256','job_id','source_sha')},
        **{'OS_EXIT.json':dict(returncode=0,foreground_wait_completed=True,finished_epoch=when+1,source_sha='d'*40),
           'CONTINUATION_PROCESS.json':dict(science_source_sha=row['source_sha'],controller_source_sha='d'*40,pid=999,procstart='123')})
    return dict(g=g,local=lb,exports=exports,settled=settled,lease=lease,observation=observation,now=when+5)


def proposal(future,arm):
    f=future;gb=e.raw(f['g']);source=f['exports'][arm]
    obs=dict(f['observation'],GLOBAL_sha256=e.sha(gb))
    small=e.raw(dict(schema='ARX5_MINIMAL_CLOSED512_INFERENCE_SMALL_READBACK_V1',arm=arm,
        GLOBAL_sha256=e.sha(gb),tensor_payloads_read=False,official_root=source['root'],
        files={n:dict(base64=base64.b64encode(source['files'][n]).decode(),bytes=len(source['files'][n]),sha256=e.sha(source['files'][n]))
            for n in transfer.SMALL+('CHECKPOINT_COMPLETE.json',)}))
    plan=transfer.prepare(gb,f['lease'],f['settled'],obs,small,arm=arm);identity=d.model_identity(plan)
    manifest=e.raw(dict(schema=d.MANIFEST,candidate=arm,checkpoint_manifest_sha256=identity['checkpoint_manifest_sha256'],
        model_transfer_only=True,CPU_upper_s=600,walltime_upper_s=7200,source_preparation_CPU_upper_s=300,
        scientific_effect_limits=dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0),
        source_sha='a'*40,parent_manifest_sha256='b'*64,compiled_plan_sha256=e.sha(e.raw(plan)),worker_sha256='c'*64,
        canonical_parent='/fixture/control-parent',control_root='/fixture/control-parent/control/upload-'+arm,
        run_uuid='upload-'+arm,checkpoint_root=source['root'],repo_id='leoyuer/pi0294-public-artifacts',
        HF_prefix='arx5-first512/'+arm+'/'+identity['checkpoint_manifest_sha256']))
    return d.propose(gb,f['local'],lease_bytes=f['lease'],settled_bytes=f['settled'],owner_observation=obs,
        small_bytes=small,arm=arm,source_manifest_bytes=manifest,now_epoch=f['now']),manifest


def test_two_source_slices_keep_actual_history_package_all14_and_all_Core_escrows(future):
    original=deepcopy(future['g']);f=deepcopy(future)
    baseline=e.scope.first512_forecast(e.raw(f['g']),f['local']);assert len(baseline['dimensions'])==14
    for arm in e.training.ARMS:
        before=deepcopy(f['g']);out,manifest=proposal(f,arm);f['g']=out['proposed_global']
        assert f['g'][e.package.FIELD]==original[e.package.FIELD]
        assert f['g']['PBS_submissions']==original['PBS_submissions']
        assert f['g'][e.original.PACKAGE]==original[e.original.PACKAGE]
        assert f['g']['CPU_s']==before['CPU_s']+900
        assert f['g']['CPU_reserved_s']==before['CPU_reserved_s']-900
        assert not out['full_future_proof']['cap_changes_required']
        assert all(out['full_future_proof']['dimensions'][k]['projected']==v['projected']for k,v in baseline['dimensions'].items())
        admitted=d.upload_admission(f['g'],out['compiled_plan'],now_epoch=f['now']+1,
            checkpoint_complete_bytes=f['exports'][arm]['files']['CHECKPOINT_COMPLETE.json'],
            source_control_manifest_bytes=manifest,source_checkpoint_root=f['exports'][arm]['root'])
        assert admitted['source_CPU_upper_s']==600 and not admitted['model_or_physics_admission']
        for change in('manifest','checkpoint','root','expired'):
            args=dict(now_epoch=f['now']+1,checkpoint_complete_bytes=f['exports'][arm]['files']['CHECKPOINT_COMPLETE.json'],
                source_control_manifest_bytes=manifest,source_checkpoint_root=f['exports'][arm]['root'])
            if change=='manifest':args['source_control_manifest_bytes']=manifest+b' '
            elif change=='checkpoint':args['checkpoint_complete_bytes']+=b' '
            elif change=='root':args['source_checkpoint_root']+='/other'
            else:args['now_epoch']=r.basic(f['g'])[1]['deadline_epoch']
            with pytest.raises(PermissionError):d.upload_admission(f['g'],out['compiled_plan'],**args)
        with pytest.raises(PermissionError):proposal(f,arm)
    first=d.budget_LOCAL(f['g'],json.loads(f['local']));assert d.budget_LOCAL(f['g'],first)==first
    assert set(d.validate(f['g']))==set(e.training.ARMS)


def test_actual_running_capture_cannot_prepay_or_create_upload_permission(captured):
    g=json.loads(captured[0]);assert r.basic(g)[1]['status']==e.STATES[1]
    with pytest.raises(PermissionError):transfer.closed_endpoint(captured[0],b'{}',b'{}',{},arm=e.training.ARMS[0])
