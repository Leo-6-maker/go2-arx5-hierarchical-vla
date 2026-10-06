"""Read-only actual schedule, S4 math, gripper supervision and full Trainers."""
import ast,importlib.util,json,math
from collections import Counter
from pathlib import Path
from train import ROOT,CONTROL,read,read_lines,sha,seed_for,STATS_SHA
from run import write

def graph(path):
    source=CONTROL/'code/training_audit.py'
    spec=importlib.util.spec_from_file_location('pi0276_qualified_graph',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='graph')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value in (10240,5120,'PI0273'):
            n.value={10240:16384,5120:8192,'PI0273':'PI0276'}[n.value]
    scope=vars(m).copy();scope['ROOT']=ROOT
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
    return scope['graph']('TASK',path)

def main():
    assert read(ROOT/'TASK/canary_compare.json')['status']=='PASS_PI0276_REAL_EXACT_RESUME_CANARY'
    for r in read(ROOT/'EXECUTION_FILESET.json')['files']:assert sha(r['path'])==r['sha256']
    p=ROOT/'TASK/main/training';samples=read_lines(p/'sample_audit.jsonl');packets=read_lines(p/'step_rng.jsonl');s4=graph(p)
    plan=read(ROOT/'design/TRAIN_PLAN.json')['stream'];proof={r['uid']:r for r in read(ROOT/'design/PROCESSED_INPUTS.json')['rows']}
    assert len(samples)==len(packets)==len(s4)==len(plan)==16384
    for i,(a,b,c,v) in enumerate(zip(samples,packets,s4,plan)):
        assert all(a[k]==v[k] for k in v)
        assert (a['unit'],a['micro'],a['group'])==(i//2+1,i%2,i%4)
        for k in ('condition_sha256','labels_masks_sha256'):assert a[k]==proof[v['uid']][k]
        assert b['seed']==seed_for(a['unit']) and b['mode_restored'] and math.isfinite(b['loss'])
        for k in ('unit','micro','condition_sha256'):assert a[k]==b[k]
        for k in ('native_epsilon','native_tau','stock_initial_noise'):assert b[k]==c[k]
        if i%2:
            for k in ('native_epsilon','native_tau','stock_initial_noise'):assert b[k]==packets[i-1][k]
    grads=read_lines(p/'gripper_output_gradients.jsonl');native=read_lines(p/'gripper_native_loss.jsonl');generated=read_lines(p/'gripper_generated_error.jsonl')
    assert len(native)==len(generated)==16384 and any(r['norm']>0 for r in grads)
    assert all(math.isfinite(r['norm']) for r in grads)
    assert all(math.isfinite(r['native_velocity_MSE']) for r in native)
    assert all(math.isfinite(r['command_fraction_MAE']) for r in generated)
    assert read(p/'HEALTH128.json')['status']=='PASS_CONTINUING_TASK_LEARNING_HEALTH'
    checkpoints=[]
    for step in (4096,8192):
        cp=p/f'checkpoint-{step}';state=read(cp/'trainer_state.json');assert (state['global_step'],state['max_steps'])==(step,8192)
        assert all((cp/n).is_file() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'))
        assert sha(cp/'effective_statistics.json')==STATS_SHA and read(cp/'statistics.json')==read(ROOT/'contracts/TASK/source/dataset/statistics.json')
        assert read(cp/'PI0276_READOUT_CONTRACT.json')==read(ROOT/'design/TASK_READOUT_R1.json')
        for n in ('EXECUTION_FILESET.json','PARAMETER_SCOPE.json','DONOR_PARAMETER_SCOPE.json','design/FITTING_CARD.json','STORAGE_ADMISSION.json','evidence/PRECISION_CONTRACT_RESOLUTION.json'):
            assert sha(cp/'pi0276_provenance'/n)==sha(ROOT/n)
        entries=[]
        for q in sorted(cp.rglob('*')):
            assert not q.is_symlink()
            if q.is_file():entries.append(dict(path=q.relative_to(cp).as_posix(),bytes=q.stat().st_size,sha256=sha(q)))
        checkpoints.append(dict(arm='TASK',step=step,path=str(cp),complete_trainer=True,entries=entries))
    updates=read_lines(ROOT/'raw_completed_updates.jsonl');heads=read_lines(ROOT/'raw_training_heads.jsonl')
    for rows,cap in ((updates,8464),(heads,155648)):
        assert len(rows)<=cap and [r['ordinal'] for r in rows]==list(range(1,len(rows)+1))
    counts=Counter(r['phase'] for r in updates);assert counts['main']==8192 and counts['continuous']>=8 and counts['split']>=8
    write(ROOT/'evidence/MAIN_TRAINING_AUDIT.json',dict(status='PASS_PI0276_ACTUAL8192_TASK_TRAINING',complete_checkpoints=checkpoints,
        raw_completed_updates=len(updates),technical_completed_updates=len(updates)-8208,raw_training_heads=len(heads),
        phase_group_consumption=dict(Counter(a['group'] for a in samples)),
        episode_consumption=dict(Counter(a['seed'] for a in samples)),native_gripper_supervision=True,
        logs={f.name:sha(f) for f in p.glob('*.jsonl')},model_queries_by_audit=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))

if __name__=='__main__':main()
