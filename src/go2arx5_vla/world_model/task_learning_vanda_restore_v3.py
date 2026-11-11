"""Independent A40 FULL_ACTION_SIDE restore qualification, not SFT2001/FIT4.

Only admission, initialization and comparison differ. The existing native
Trainer, FM loss, optimizer, sampler, noise audit and site kernel are reused.
This entry has no permission to start a500-update arm or physical execution.
"""
import json,math
from pathlib import Path
from . import target_gr00t_profile_v1 as original
from .storage import digest

PROFILE='TASK_LEARNING_VANDA_FULL_ACTION_RESTORE_V3'
KIND='SFT2000_INITIALIZED_FULL_ACTION_NATIVE_RESTORE_QUALIFICATION_V3'
DATA='ff94fc77c5a8309dc1460032b308c2f213c96c088449ee1821b3e46e045107b0'
STATS='0faba237b8922ed9bbd53bd6732a9f449f46dcb6fc19017eae8f3cffa3afd9f2'
PACKAGE='7feffa8dd11b5c9ce963f2e0f2f2613196c1a3ae8e5039056377bd5e25e02250'


def recipe(dataset_sha256,statistics_sha256,*,endpoint):
    original.need(dataset_sha256==DATA and statistics_sha256==STATS,
        'VANDA_V1_SEALED_TRAIN100_NOT_OLD_FIT4')
    original.need(type(endpoint)is int and endpoint in(2,4,5),'VANDA_V1_FINITE_QUALIFICATION_ONLY')
    p=original.recipe(DATA,STATS,mode='RESTORE4',variant='FULL_ACTION_SIDE',endpoint=2)
    p.update(schema='TASK_LEARNING_VANDA_RESTORE_RECIPE_V3',profile=PROFILE,kind=KIND,
        endpoint=endpoint,seed=20261109,initial_package_sha256=PACKAGE,
        initial_optimizer='FRESH_ADAMW_FROM_SEALED_SFT2000_WEIGHTS',
        scheduler='COSINE_FIXED_8000_WARMUP200',scheduler_horizon=8000,warmup_updates=200,
        adaptation_step_zero_is_SFT2000=True,qualification_only=True,
        open_loop_probes='FIXED_REAL_PHASES_NORMAL_AND_RECOVERY_TRAIN40',
        next_endpoint='NO_AUTOMATIC500_CORE_C_MECHANISM_GATE_NEGATIVE')
    return p


def validate(spec):
    p=spec.get('recipe')or{};need=original.need
    expected=recipe(p.get('dataset_sha256'),p.get('statistics_sha256'),endpoint=p.get('endpoint'))
    start=spec.get('start_step');init=spec.get('initial_package')or{}
    need(p==expected and spec.get('profile')==PROFILE and spec.get('kind')==KIND
        and spec.get('execution_site')=='VANDA_A40'and spec.get('physics')is False
        and spec.get('WBC_changes')is False and type(start)is int
        and(start,p['endpoint'])in((0,2),(0,4),(2,4),(4,5))
        and spec.get('GR00T_updates')==p['endpoint']-start
        and spec.get('WM_updates')==spec.get('Actor_updates')==spec.get('TASK_ADAPTER_updates')==0
        and spec.get('parent_manifest_sha256')==original.PARENT
        and set(init)=={'path','sha256'}and init['sha256']==PACKAGE
        and Path(init['path']).is_absolute()
        and all(not spec.get(k)for k in('fusion_binding','pilot1_binding','next1_binding','next2_binding','night1_binding')),
        'VANDA_V1_NEW_FULL_ACTION_SFT2000_GENERATION_AND_ACTUAL_SITE')
    if start:
        from .task_learning_training_profile_v1 import descriptor
        need(descriptor(spec.get('resume')),'VANDA_V1_COMPLETE_SAME_GENERATION_RESUME')
    else:need(spec.get('resume')is None,'VANDA_V1_FRESH_OPTIMIZER_AT_ZERO')
    need(type(spec.get('deadline_epoch'))in(int,float)and math.isfinite(spec['deadline_epoch']),
        'VANDA_V1_FINITE_DEADLINE')
    return p


def compare_trace(first,second,*,start,stop):
    """Compare consumed batches/noise/RNG/loss and actual clipped gradients."""
    need=original.need
    def rows(root,name,key):
        values=[json.loads(v)for v in(Path(root)/name).read_text().splitlines()]
        ids=[v[key]for v in values]
        need(len(ids)==len(set(ids)),'VANDA_V1_NO_DUPLICATE_TRACE')
        return {v[key]:v for v in values}
    a=rows(first,'evidence/MICROBATCHES.jsonl','cursor')
    b=rows(second,'evidence/MICROBATCHES.jsonl','cursor')
    cursors=list(range(start*32,stop*32))
    need(all(c in a and c in b and a[c]==b[c]for c in cursors),
        'VANDA_V1_EXACT_NATIVE_BATCH_NOISE_LOSS_RNG')
    ga=rows(first,'GRADIENTS_ACTUAL.jsonl','step')
    gb=rows(second,'GRADIENTS_ACTUAL.jsonl','step')
    need(all(n in ga and n in gb and ga[n]==gb[n]for n in range(start+1,stop+1)),
        'VANDA_V1_EXACT_ACTUAL_CLIPPED_GRADIENTS')
    return dict(exact=True,consumed_microbatches=len(cursors),actual_optimizer_steps=stop-start,
        batch_noise_time_RNG_loss_and_all_clipped_gradients=True)


def gradient_callback(root,contract):
    """Read-only audit after the original clipping callback; no storage offload."""
    from transformers import TrainerCallback
    from scripts.pi0294.native_train import tensor_identity
    original.need(contract.describe()['trainable_numel']==1304913028,
        'VANDA_V1_ACTUAL_FULL_ACTION_PARAMETER_COUNT')
    class GradientAudit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            values={n:tensor_identity(model.get_parameter(n).grad)if model.get_parameter(n).grad is not None else None
                for n in contract.trainable_names}
            with(Path(root)/'GRADIENTS_ACTUAL.jsonl').open('a')as handle:
                handle.write(json.dumps(dict(step=state.global_step+1,all_declared_gradients=values),sort_keys=True)+'\n')
    return GradientAudit()


def run(c):
    from .target_gr00t_allocation_v1 import with_allocation,compare_next_update
    from scripts.pi0292.durable import write
    original.need(c.get('profile')==PROFILE and c.get('initial_package',{}).get('sha256')==PACKAGE,
        'VANDA_V1_EXPLICIT_ALLOCATION_AND_INITIAL_WEIGHTS')
    def qualify(root,need,deadline,gpu,child,children,started,stop_requested):
        import time
        # Same finite training allocation; no extra GPU job for CPU preparation.
        # The new run owns this independent data copy, never the CLOSED transfer.
        from scripts.pi0294.e1.journal import Journal
        from scripts.wm02.qualify_target_gr00t_processor_v1 import qualify as processor_qualify
        from .target_gr00t_delivery_v1 import verify
        verify(c['initial_package']['path'],PACKAGE,c['parent']['files'])
        j=Journal(root/'control/journal',root.name)
        def processor():
            import torch
            torch.set_num_threads(4)
            need(not torch.cuda.is_initialized(),'VANDA_V1_PROCESSOR_BEFORE_CUDA')
            value=processor_qualify(c['bindings']['dataset'],c['bindings']['source'],c['bindings']['cosmos'],
                verify_transfer=True,transfer_site='VANDA_A40',dataset_family='TASK_LEARNING_CONTINUOUS_V1')
            need(not torch.cuda.is_initialized(),'VANDA_V1_NO_GPU_PROCESSOR_COMPUTE')
            return {k:v for k,v in value.items()if k!='records'}
        proof=j.once('CONTINUOUS_ORIGINAL_PROCESSOR_RECONSTRUCTION',dict(dataset=DATA,statistics=STATS),processor)
        write(root/'PROCESSOR_RECONSTRUCTION_ACTUAL.json',proof)
        generation=c['generation_id']+'-fresh-SFT2000-restore'
        def execute(name,endpoint,*,start=0,checkpoint=None):
            row=child(name,'RESTORE4','FULL_ACTION_SIDE',endpoint,start_step=start,
                resume=None if checkpoint is None else dict(path=checkpoint,sha256=digest(Path(checkpoint)/'CHECKPOINT_COMPLETE.json')),
                generation=generation)
            need(row['exit_code']==0 and row['endpoint_complete'],'VANDA_V1_CHILD_FAILED_'+name)
            return row
        full=execute('continuous4',4);split=execute('split2',2)
        resumed=execute('resume4',4,start=2,checkpoint=split['checkpoint'])
        state4=compare_next_update(full['checkpoint'],resumed['checkpoint'])
        trace02=compare_trace(full['root'],split['root'],start=0,stop=2)
        trace24=compare_trace(full['root'],resumed['root'],start=2,stop=4)
        write(root/'FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(state4=state4,trace02=trace02,trace24=trace24))
        next_full=execute('continuous-next5',5,start=4,checkpoint=full['checkpoint'])
        next_resumed=execute('resumed-next5',5,start=4,checkpoint=resumed['checkpoint'])
        state5=compare_next_update(next_full['checkpoint'],next_resumed['checkpoint'])
        trace45=compare_trace(next_full['root'],next_resumed['root'],start=4,stop=5)
        proof=dict(state4=state4,state5=state5,trace02=trace02,trace24=trace24,trace45=trace45)
        write(root/'RESTORE_AND_ACTUAL_NEXT_UPDATE_ACTUAL.json',proof)
        return dict(status='PASS_ACTUAL_VANDA_V1_FULL_ACTION4_VS2_PLUS2_AND_NEXT_UPDATE',
            children=children,proof=proof,actual_optimizer_calls=sum(v['GR00T_updates']for v in children),
            actual_sample_visits=sum(v['GR00T_updates']for v in children)*32,
            actual_model_loads=sum(v['actual_model_loads']for v in children),
            initial_SFT2000_package_sha256=PACKAGE,dataset_sha256=DATA,statistics_sha256=STATS,
            qualification_weights_used_for_candidate=False,formal500_started=False,
            Core_C_mechanism='NEGATIVE_NO_AUTOMATIC_V3',native_integrations=0,wall_s=time.monotonic()-started)
    return with_allocation(c,qualify)
