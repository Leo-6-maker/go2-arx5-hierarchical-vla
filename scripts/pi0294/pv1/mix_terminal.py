"""Independent post-F state audit around original streaming/member checks."""
import json,os,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1.readonly_checkpoint_hash import StreamHashes
from . import mix1024 as mix
from .recipe import PARENT
from scripts.pi0294.e1.trainer import check_trace


def audit(root,request):
    root=Path(root);arm=request['arm'];cp=root/'training'/arm/'checkpoint-1024';full=read(cp/'CHECKPOINT_COMPLETE.json');before=sha(cp/'CHECKPOINT_COMPLETE.json');members=full['files'];stream=StreamHashes();required={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json','config.json'}
    if (full['global_step']!=1024 or full['full_resume']is not True or not required<=set(members)
        or full['origin_job_id']!=request['job_id'] or full['origin_source_sha']!=request['source_sha']):raise PermissionError('V5_EXACT_CURRENT_JOB_COMPLETE_STATE')
    for n,h in members.items():
        p=cp/n
        if p.is_symlink() or not p.resolve().is_relative_to(cp.resolve()) or stream(p)!=h:raise PermissionError('V5_FULL_STATE_MEMBER_CHANGED:'+n)
    original=read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json');recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][arm];mix.validate(recipe,original);saved=read(cp/'SAMPLER_STATE.json');state=read(cp/'trainer_state.json')
    from .dataset import PV1Dataset
    b=read(root/'NEURAL_BINDINGS.private.json');i=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    data=PV1Dataset(b['dataset'],b['source'],b['cosmos'],for_optimization=True,raw_bindings=read(b['raw_bindings']),correction_root=i['C1_dataset'],correction_bindings=read(i['C1_raw_bindings']),new_root=i['new_dataset'],new_raw_bindings=read(i['new_raw_bindings']),route='COVERAGE_FIRST',arm='B_COVERAGE',seed=20261019)
    data=mix.bind_dataset_reader(data,arm);expected=data.make_sampler(data).state_dict(mix.VISITS)
    if saved['sampler']!=expected or saved['recipe']!=recipe or saved['manifest_sha256']!=data.manifest_sha or state['global_step']!=1024 or stream(cp/'statistics.json')!=mix.STATS:raise PermissionError('V5_COMPLETE_RECIPE_CURSOR_R0_AND_SAMPLER')
    session=root/'training'/arm/('session-'+request['request_id']);trace=check_trace(session/'worker_trace',data,0,mix.VISITS)
    # CPU-only actual serialization inspection; no model or CUDA operation.
    import torch
    opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True);sched=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False);rng=torch.load(cp/'rng_state.pth',map_location='cpu',weights_only=False)
    if not opt.get('state') or not opt.get('param_groups') or sched.get('last_epoch')!=1024 or not {'python','numpy','cpu'}<=set(rng):raise PermissionError('V5_REAL_OPTIMIZER_SCHEDULER_AND_RNG_STATE')
    steps=set()
    for row in opt['state'].values():
        step=row.get('step');steps.add(int(step.item() if hasattr(step,'item') else step))
        for name in ('exp_avg','exp_avg_sq'):
            if name not in row or not bool(torch.isfinite(row[name]).all()):raise ValueError('V5_FINITE_ACTUAL_OPTIMIZER_MOMENTS')
    if steps!={1024}:raise PermissionError('V5_OPTIMIZER_STATE_MUST_BE_1024')
    probes=root/'evidence'/(arm+'-1024')/'TRAIN_PROBES';p=read(probes/'RESULT.json')
    if p.get('complete')is not True or p['checkpoint_manifest_sha256']!=before or len(p['records'])!=12 or p['new_physics']!=0:raise PermissionError('V5_ORIGINAL_FIXED_TWELVE_TRAIN_PROBES')
    for row in p['records']:
        if stream(probes/(row['id']+'.npz'))!=row['arrays_sha256']:raise PermissionError('V5_PROBE_BYTES_CHANGED')
    witness=stream.witnesses()
    if sha(cp/'CHECKPOINT_COMPLETE.json')!=before:raise PermissionError('V5_MANIFEST_CHANGED_DURING_INDEPENDENT_AUDIT')
    return dict(status='PASS_INDEPENDENT_V5_CD_1024_COMPLETE_STATE',arm=arm,checkpoint_manifest_sha256=before,checkpoint_root=str(cp),member_count=len(members),actual_consumed_cursor=32768,parent_checkpoint_manifest_sha256=PARENT,statistics_sha256=mix.STATS,sampler=saved['sampler'],trace=trace,optimizer_steps=sorted(steps),scheduler_last_epoch=sched['last_epoch'],rng_keys=sorted(rng),streamed_witness=witness,probe_result_sha256=sha(probes/'RESULT.json'),new_model_loads=0,new_queries=0,new_updates=0,new_physics=0)
