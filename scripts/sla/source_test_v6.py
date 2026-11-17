"""Once-frozen source TEST materialization/evaluation, never source selection."""
import json,shutil,time,traceback
from pathlib import Path
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin


def admission(spec):
    train=read_pin(spec['frozen_training_result']);campaign=read_pin(spec['source_campaign_result']);training_spec=read_pin(spec['training_spec'])
    if(spec['plan_commit']!='ae3d3c0dceade437b0104a150cd0402fef650052' or spec['physics']or spec['new_physical_requests_budget']
       or spec['new_optimizer_updates_budget']or train['exit_code']or train['new_updates']!=1000 or train['new_visits']!=32000
       or campaign['exit_code']or campaign['new_updates']!=1000 or len(campaign['records'])!=3 or not train['full_restore']['bit_exact']):
        raise PermissionError('POST_FIXED1000_FROZEN_SOURCE_TEST_ONLY')
    first=campaign['records'][0]
    if(first['result_path']!=spec['frozen_training_result']['path']or first['result_sha256']!=spec['frozen_training_result']['sha256']
       or Path(training_spec['registered_root'])!=Path(spec['frozen_training_result']['path']).parent
       or train['source_sha']!=training_spec['source_sha']or sha(Path(train['last_checkpoint']['path'])/'COMPLETE.json')!=train['last_checkpoint']['sha256']):
        raise ValueError('ONE_ACTUAL_FROZEN_SOURCE_TRAINING_AND_RESTORE_LINEAGE_REQUIRED')
    for record in campaign['records'][1:]:
        result=read_pin(dict(path=record['result_path'],sha256=record['result_sha256']))
        if not result['passed']or result['actual']['ordinary_RGB_generations']!=16:raise ValueError('INDEPENDENT_SOURCE_RGB16_RESTORE_REQUIRED')
    return train,training_spec


def prepare(spec):
    from go2arx5_vla.sla_adapter.source_native_v6 import load_numeric,anchor_strata,SCHEMA,COORDINATES
    from scripts.sla.source_prepare_v6 import evaluation_bank
    train,training_spec=admission(spec);root=Path(spec['registered_root']);start=time.monotonic();dataset=Path(training_spec['dataset'])
    manifest=read_pin(dict(path=str(dataset/'MANIFEST.private.json'),sha256=training_spec['source_manifest_sha256']))
    stats_path=dataset/manifest['statistics']['file'];stats=read_pin(dict(path=str(stats_path),sha256=manifest['statistics']['sha256']))
    if spec['new_model_generations_budget']or spec['new_training_visits_budget']:raise PermissionError('CPU_TEST_MATERIALIZATION_ZERO_MODEL_EFFECTS')
    output=root/'dataset';output.mkdir();(output/'labels').mkdir();shutil.copyfile(stats_path,output/'STATISTICS.json');episodes=[]
    if len(manifest['source_TEST'])!=12:raise ValueError('EXACT_FROZEN_SOURCE_TEST12_REQUIRED')
    for record in manifest['source_TEST']:
        if time.time()>=spec['deadline_epoch']-120 or(root/'STOP.json').exists():raise TimeoutError('FINITE_TEST_PREPARATION_LIMIT')
        source=record['source'];raw=Path(record['raw_path'])
        if sha(raw)!=source['sha256']:raise ValueError('ORIGINAL_TEST_RAW_SHA_CHANGED')
        state,actions,text,audit=load_numeric(raw);dynamic=anchor_strata(state,actions,stats['state']['IQR'])
        baseline=np.zeros_like(state);baseline[:,6]=state[:,6];baseline[:,13]=state[:,13]
        future=np.any(np.max(abs(actions-baseline[:,None,:]),axis=1)>.01*np.asarray(stats['actions']['IQR']),axis=1)
        eid=source['task']+'-'+Path(source['path']).stem;path=output/'labels'/(eid+'.npz');np.savez(path,state=state,actions=actions,dynamic=dynamic,future_span_dynamic=future)
        episodes.append(dict(id=eid,task=source['task'],split='SOURCE_TEST',valid=True,windows=len(state),instruction=text,raw_path=str(raw),raw_sha256=source['sha256'],audit=audit,
            arrays=dict(file=str(path.relative_to(output)),sha256=sha(path)),_dynamic=dynamic))
        write(root/'PROGRESS.private.json',dict(phase='POST_FREEZE_SOURCE_TEST_NUMERIC',episodes=len(episodes),new_generations=0,new_updates=0,new_native_requests=0))
    bank=evaluation_bank(episodes,'SOURCE_TEST');write(output/'TEST96_BANK.private.json',bank)
    for ep in episodes:ep.pop('_dynamic')
    result=dict(schema=SCHEMA,coordinates=COORDINATES,horizon=50,action_dimensions=14,padding_dimensions=18,label_kind='FUTURE_LOGGED_STATE_REFERENCE',episodes=episodes,
        statistics=dict(file='STATISTICS.json',sha256=sha(output/'STATISTICS.json')),evaluation_bank=dict(file='TEST96_BANK.private.json',sha256=sha(output/'TEST96_BANK.private.json')),
        frozen_training_result=spec['frozen_training_result'],source_campaign_result=spec['source_campaign_result'],TRAIN_statistics_refitted=False,
        source_train_manifest_sha256=training_spec['source_manifest_sha256'],test_model_endpoint=train['last_checkpoint'],physical_calibration='UNKNOWN',test_opened_after_fixed_endpoint=True)
    write(output/'MANIFEST.private.json',result)
    return dict(exit_code=0,status='POST_FREEZE_TEST12_TYPED_READY_NO_MODEL_GENERATIONS',dataset=dict(path=str(output),manifest_sha256=sha(output/'MANIFEST.private.json')),
        statistics_sha256=sha(output/'STATISTICS.json'),test_bank=dict(path=str(output/'TEST96_BANK.private.json'),sha256=sha(output/'TEST96_BANK.private.json')),
        actual_episodes=12,fixed_test_windows=96,new_updates=0,new_visits=0,new_generations=0,new_native_requests=0,wall_s=time.monotonic()-start)


def evaluate(spec):
    from go2arx5_vla.sla_adapter.source_training_v6 import SourceEngine
    from go2arx5_vla.sla_adapter.source_native_v6 import SourceTestDataset
    from go2arx5_vla.sla_adapter.source_evaluation_v6 import evaluate as score
    from scripts.sla.train_source_v6 import verify
    from openpi_client.image_tools import resize_with_pad
    train,training_spec=admission(spec);root=Path(spec['registered_root']);prepared=read_pin(spec['test_preparation_result'])
    if prepared['exit_code']or prepared['new_generations']or spec['new_model_generations_budget']!=96:raise ValueError('ONE_FROZEN_TEST96_PANEL_REQUIRED')
    # Restore through the exact already-trained model implementation, not an
    # undocumented loader revision carrying the same optimizer metadata.
    for name in('source_training_v6.py','model_v1.py','source_batch_v6.py'):
        relative='src/go2arx5_vla/sla_adapter/'+name
        if sha(Path(spec['source_root'])/relative)!=training_spec['source_files'][relative]:raise ValueError('SOURCE_TRAINED_MODEL_RESTORE_IMPLEMENTATION_CHANGED')
    engine=SourceEngine(training_spec,root/'work')
    try:
        engine.resume(train['last_checkpoint']);full=verify(engine,train['last_checkpoint'])
        d=SourceTestDataset(prepared['dataset']['path'],prepared['dataset']['manifest_sha256'],frozen_training_result=spec['frozen_training_result'],
            source_campaign_result=spec['source_campaign_result'],resize_with_pad=resize_with_pad)
        bank=read_pin(prepared['test_bank']);summary=score(engine,root,'SOURCE_TEST',endpoint='S1000',dataset=d,windows=bank)
        if engine.counts['ordinary_RGB_generations']!=96:raise ValueError('EXACT96_ONCE_ONLY_TEST_GENERATIONS_REQUIRED')
        return dict(exit_code=0,status='FROZEN_SOURCE_TEST96_COMPLETE_NOT_TASK_SUCCESS',source_sha=spec['source_sha'],frozen_training_result=spec['frozen_training_result'],
            SOURCE_TEST96=summary,full_restore=full,actual=engine.counts,new_updates=0,new_visits=0,new_generations=96,new_native_requests=0,
            transfer_selection_changed_after_TEST=False,source_task_success_rate=None)
    finally:engine.close()


def main(spec,*,gpu=False):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    root=Path(spec['registered_root']);lock=None
    try:
        if gpu:
            from scripts.sla.close_goal_g1_v1 import wait_resource_owner
            from go2arx5_vla.sla_adapter.site_v2 import validate
            lock=wait_resource_owner(spec);validate(spec)
        result=evaluate(spec)if gpu else prepare(spec)
    except BaseException:result=dict(exit_code=1,status='PRESERVED_SOURCE_TEST_TECHNICAL_FAILURE_NO_AUTORETRY',failure=traceback.format_exc(),new_updates=0,new_native_requests=0)
    finally:
        if lock:lock.close()
    write(root/'RESULT.private.json',result);close(spec,result['exit_code']);return result['exit_code']
