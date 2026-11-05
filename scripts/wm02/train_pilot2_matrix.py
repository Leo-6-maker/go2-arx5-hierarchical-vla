"""One admitted local GPU worker for the finite, frozen PILOT2 factorial study."""
import argparse
import gc
import json
import time
from pathlib import Path

from go2arx5_vla.world_model.actor_sampling import MODES

PLAN_SHA='6d82fa23c892f167001abe743dd5eadfca6fd777'


def validate_design(spec):
    arms=spec['study_arms']
    expected={(d,m,s)for d in ('OLD_24','EXPANDED_53')for m in MODES for s in (36001,36002,36003)}
    if (spec.get('pilot2_plan_sha')!=PLAN_SHA or len(arms)!=12
            or {(a['dataset'],a['sampler'],a['seed'])for a in arms}!=expected
            or spec.get('endpoint')!=96000 or spec.get('arm_endpoint')!=8000
            or spec.get('batch_size')!=128 or spec.get('learning_rate')!=.001
            or spec.get('physics')is not False or spec.get('kind')!='ACTOR_STUDY'):
        raise ValueError('FROZEN_PILOT2_TWO_BY_TWO_BY_THREE_DESIGN_REQUIRED')
    for d,n in [('OLD_24',24),('EXPANDED_53',53)]:
        if spec['study_datasets'][d]['TRAIN_windows']!=n:
            raise ValueError('ORIGINAL_PHYSICAL_WINDOW_COUNT_REQUIRED')
    return arms


def audit_checkpoint(spec, dataset):
    import numpy as np
    import torch
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.pilot1_actor import Actor,examples,export_arrays,numpy_logits
    from go2arx5_vla.world_model.actor_diagnostics import grouped_diagnostics,normalizer_diagnostics
    from go2arx5_vla.world_model.storage import digest
    item=spec['study_datasets'][dataset]
    if digest(item['legacy_checkpoint'])!=item['legacy_checkpoint_sha256']:
        raise ValueError('PINNED_EXISTING_ACTOR_CHANGED')
    data,meta=load(item['path']);task=json.loads(Path(spec['frozen_task']).read_text())
    if (any(e['split'] not in ('TRAIN','DEV')for e in meta['episodes'])
            or len(select_samples(data,meta,'TRAIN'))!=item['TRAIN_windows']
            or len(select_samples(data,meta,'DEV'))!=8):
        raise ValueError('EXACT_TRAIN_WINDOWS_AND_UNCHANGED_EIGHT_DEV_REQUIRED_NO_TEST')
    with np.load(spec['ARX'],allow_pickle=False)as a:teacher=CausalARX.from_arrays(meta,a)
    saved=torch.load(item['legacy_checkpoint'],map_location='cpu',weights_only=True)
    model=Actor(saved['dimension']).to('cuda:0');model.load_state_dict(saved['model'])
    result={}
    for split in ('TRAIN','DEV'):
        ids=select_samples(data,meta,split);x,q,scores=examples(teacher,data,meta,ids,task)
        with torch.no_grad():logits=model(torch.as_tensor(x,dtype=torch.float32,device='cuda:0')).cpu().numpy()
        parents=np.repeat([meta['episodes'][int(data['episode_index'][i])]['parent']for i in ids],256)
        regions=np.tile(np.repeat(np.arange(4),64),len(ids))
        metrics=grouped_diagnostics(logits,q,scores,parents,regions,high_margin_m=task['teacher_temperature_m'])
        metrics['normalizer']=normalizer_diagnostics(x,model.mean.cpu().numpy(),model.scale.cpu().numpy())
        metrics['physical_windows']=len(ids)
        exported=numpy_logits(export_arrays(model),x[:256])
        metrics['numpy_torch_max_logit_error']=float(np.max(np.abs(exported-logits[:256])))
        metrics['numpy_torch_argmax_differences']=int(np.sum(exported.argmax(1)!=logits[:256].argmax(1)))
        result[split]=metrics
    return dict(status='A0_DISTILLATION_DIAGNOSIS',dataset=dataset,
                checkpoint_sha256=item['legacy_checkpoint_sha256'],replay_sha256=item['sha256'],
                metrics=result,previous_sampler='ORIGINAL_GROUPED_CYCLIC',
                sampler_cause='HYPOTHESIS_NOT_PROVEN',TEST_read=False,native_integrations=0,optimizer_updates=0)


def run(spec,output):
    import torch
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.pilot1_storage import local_path
    from scripts.wm02.train_pilot1_actor import run as train
    arms=validate_design(spec);output=Path(output);started=time.monotonic();cpu=time.process_time()
    for item in spec['study_datasets'].values():
        local_path(spec,item['path'])
        if digest(item['path'])!=item['sha256']:raise ValueError('FROZEN_STUDY_DATA_CHANGED')
    if digest(spec['frozen_task'])!=spec['frozen_task_sha256']or digest(spec['ARX'])!=spec['ARX_sha256']:
        raise ValueError('FROZEN_STUDY_TEACHER_OR_TASK_CHANGED')
    torch.set_num_threads(len(spec['cpus']))
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(spec['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    results=[];attempts=[];failure=None
    try:
        diagnoses=[audit_checkpoint(spec,d)for d in ('OLD_24','EXPANDED_53')]
        write_json(output/'A0_DISTILLATION_DIAGNOSIS.json',dict(arms=diagnoses,TEST_read=False))
        for i,arm in enumerate(arms):
            if time.time()>=spec['deadline_epoch']or any((Path(spec[k])/'STOP.json').exists()for k in('registered_root','campaign_root')):
                raise TimeoutError('FINITE_MATRIX_DEADLINE_OR_OPERATOR_STOP')
            dataset=spec['study_datasets'][arm['dataset']];root=output/('arm-%02d'%i);root.mkdir()
            n=dataset['TRAIN_windows']*256
            checkpoints=sorted({2000,8000,20*n//128,40*n//128})
            s=dict(spec);s.update(kind='ACTOR_DISTILLATION',actor_recipe='WM02_PILOT2_ACTOR_V1',
                replay=dataset['path'],replay_sha256=dataset['sha256'],endpoint=8000,
                checkpoints=checkpoints,curve_every=500,seed=arm['seed'],sampler=arm['sampler'])
            for key in ('resume','resume_sha256','feedback_recipe','feedback_weight'):s.pop(key,None)
            if arm['seed']==36001 and arm['sampler']==MODES[0]:
                s.update(legacy_control_checkpoint=dataset['legacy_checkpoint'],
                         legacy_control_checkpoint_sha256=dataset['legacy_checkpoint_sha256'])
            write_json(root/'ARM_SPEC.json',s)
            write_json(output/'PROGRESS.json',dict(arm_index=i,arm=arm,completed_arms=len(results),
                completed_main_updates=sum(v['result']['main_Actor_updates']for v in results),updated_epoch=time.time()))
            attempt=dict(arm_index=i,arm=arm,main_updates_logged=0,restore_probes_recorded=0,status='STARTED')
            attempts.append(attempt)
            try:
                value=train(s,root)
                attempt['status']='TRAINED_AND_RELOADABLE'
            finally:
                log=root/'OPTIMIZER_STEPS.jsonl'
                if log.exists():
                    raw=log.read_bytes();complete=raw.splitlines()
                    if raw and not raw.endswith(b'\n'):
                        complete=complete[:-1];attempt['partial_optimizer_record_preserved']=True
                    if complete:
                        attempt['main_updates_logged']=json.loads(complete[-1])['update']
                probe=root/'RESTORE_PROBE.json'
                if probe.exists():attempt['restore_probes_recorded']=json.loads(probe.read_text())['actual_optimizer_calls']
                write_json(output/'ACTUAL_ATTEMPTS.json',attempts)
            results.append(dict(arm_index=i,arm=arm,result=value,spec_sha256=digest(root/'ARM_SPEC.json')))
            write_json(output/'COMPLETED_ARMS.json',results)
            gc.collect();torch.cuda.empty_cache()
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        raise
    finally:
        result=dict(status='PILOT2_MATCHED_ACTOR_MATRIX_COMPLETE'if len(results)==12 and failure is None else'PILOT2_MATRIX_PARTIAL_FAILED_PRESERVED',
            source_sha=spec['source_sha'],pilot2_plan_sha=PLAN_SHA,results=results,attempts=attempts,failure=failure,
            main_Actor_updates=sum(v['main_updates_logged']for v in attempts),
            separate_restore_probes=sum(v['restore_probes_recorded']for v in attempts),
            WM_updates=0,native_integrations=0,TEST_read=False,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started,
            equal_exposure_epochs=[20,40],primary_update_comparisons=[2000,8000],
            future_20000_rule='FRESH_SUCCESSOR_ONLY_IF_TRAIN_HIGH_MARGIN_ERROR_GT_0_02_AND_KL_DROP_2K_TO_8K_GE_0_005_AND_DEV_REGRET_NOT_WORSE_BY_MORE_THAN_MAX_20_PERCENT_0_005MM')
        write_json(output/'RESULT.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.output.name,'wm02-pilot1-neural',a.output,gpu=True)as spec:
        print(json.dumps(run(spec,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
