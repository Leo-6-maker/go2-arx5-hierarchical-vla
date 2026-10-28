"""Actual complete EQUIL training versus the closed BROAD packets and base."""
import ast,importlib.util,math,sys
from collections import Counter
from pathlib import Path
from train import ROOT,CONTROL,read,sha,read_lines,seed_for,STATS_SHA
spec=importlib.util.spec_from_file_location('pi0273_closed_training_audit',CONTROL/'code/training_audit.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.ROOT=ROOT
node=next(n for n in ast.parse((CONTROL/'code/training_audit.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='graph')
for n in ast.walk(node):
    if isinstance(n,ast.Constant) and n.value=='PI0273':n.value='PI0274'
scope=vars(old).copy();scope['ROOT']=ROOT;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(CONTROL/'code/training_audit.py'),'exec'),scope);graph=scope['graph']
def main():
    assert read(ROOT/'EQUIL/canary_compare.json')['status']=='PASS_PI0274_REAL_EXACT_RESUME_CANARY'
    cpu=read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json');assert cpu['status']=='PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE'
    seal=read(ROOT/'EXECUTION_FILESET_R1.json')
    for r in seal['files']:assert sha(r['path'])==r['sha256']
    path=ROOT/'EQUIL/main/training';samples=read_lines(path/'sample_audit.jsonl');packets=read_lines(path/'step_rng.jsonl');s4=graph('EQUIL',path)
    assert len(samples)==len(packets)==10240 and samples==read_lines(ROOT/'evidence/input-audit/EQUIL/sample_audit.jsonl')
    control_path=CONTROL/'BROAD/main/training/step_rng.jsonl';assert sha(control_path)=='c7306e4bd9f14dcaee9203e12f6274f612b4d731b1e0aaddd3e963186d13bdb3';control=read_lines(control_path)
    for i,(a,b,c,d) in enumerate(zip(samples,packets,s4,control)):
        assert (a['unit'],a['micro'])==(i//2+1,i%2)
        for k in ('unit','micro','command','condition_sha256'):assert a[k]==b[k]
        for k in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'):assert b[k]==d[k]
        for k in ('native_epsilon','native_tau','stock_initial_noise'):assert b[k]==c[k]
        assert b['seed']==seed_for(a['unit']) and b['mode_restored'] and math.isfinite(b['loss'])
    base=[r for r in samples if r['stream']=='BASE'];supp=[r for r in samples if r['stream']=='SUPPLEMENT'];assert [r['source_index'] for r in base]==list(range(8192)) and len(supp)==2048 and len({r['pose_id'] for r in supp})==672
    for phase in ('continuous','split','main'):
        kernel=read(ROOT/'EQUIL'/phase/'training/CPU_ENCODER_KERNEL.json');assert kernel['core']=='Prescott' and kernel['library_sha256']==cpu['kernel_sha256']
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==STATS_SHA;checkpoints=[]
    for step in (2560,5120):
        cp=path/f'checkpoint-{step}';state=read(cp/'trainer_state.json');assert (state['global_step'],state['max_steps'])==(step,5120)
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'):assert (cp/name).is_file()
        assert read(cp/'statistics.json')==read(stats) and sha(cp/'effective_statistics.json')==STATS_SHA
        assert read(cp/'PI0274_READOUT_CONTRACT.json')==read(ROOT/'design/EQUIL_READOUT_R1.json')
        for name in ('EXECUTION_FILESET_R1.json','PARAMETER_SCOPE.json','DONOR_PARAMETER_SCOPE.json','design/FITTING_CARD.json','STORAGE_ADMISSION.json','evidence/PRECISION_CONTRACT_RESOLUTION.json'):
            assert sha(cp/'pi0274_provenance'/name)==sha(ROOT/name)
        assert sha(cp/'pi0274_provenance/CPU_ENCODER_KERNEL.json')==sha(path/'CPU_ENCODER_KERNEL.json')
        entries=[]
        for p in sorted(cp.rglob('*')):
            assert not p.is_symlink()
            if p.is_file():entries.append(dict(path=p.relative_to(cp).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
        checkpoints.append(dict(arm='EQUIL',step=step,path=str(cp),entries=entries,complete_trainer=True))
    updates=read_lines(ROOT/'raw_completed_updates.jsonl');heads=read_lines(ROOT/'raw_training_heads.jsonl')
    for entries,cap in ((updates,5264),(heads,98304)):
        assert [r['ordinal'] for r in entries]==list(range(1,len(entries)+1)) and len(entries)<=cap
        assert all(r['arm']=='EQUIL' and r['phase'] in ('continuous','split','main') for r in entries)
    counts=Counter(r['phase'] for r in updates);assert counts['main']==5120 and counts['continuous']==counts['split']==8 and len(updates)==5136
    consumed=Counter((r['condition_sha256'],r['labels_masks_sha256'],r['stream']) for r in samples);early=read(CONTROL/'design/NEW_FIT192.json')['rows'];consumption=[]
    for r in early:
        key=r['condition_sha256'],r['labels_masks_sha256'];assert consumed[(*key,'BASE')]>0
        consumption.append(dict(input_id=r['input_id'],condition_sha256=key[0],labels_masks_sha256=key[1],actual_seen=True,actual_counts=dict(EQUIL=sum(consumed[(*key,s)] for s in ('BASE','SUPPLEMENT')))))
    old.archive.save(ROOT/'evidence/MAIN_TRAINING_AUDIT.json',dict(status='PASS_PI0274_ACTUAL5120_MATCHED_BROAD_TRAINING',complete_checkpoints=checkpoints,NEW_FIT192_actual_consumption=consumption,
        actual_random_packets_matched=True,actual_inputs_match_CPU_plan=True,actual_loss_gradient_mode_checked=True,actual_generated_nodes=dict(EQUIL=4),
        actual_supplement_pose_occurrences=dict(Counter(r['pose_id'] for r in supp)),exposure=[dict(task=k[0],command=k[1],bucket=k[2],microbatches=v) for k,v in sorted(Counter((r['task'],r['command'],r['bucket']) for r in samples).items())],
        logs={p.name:sha(p) for p in path.glob('*.jsonl')},raw_completed_updates=len(updates),technical_completed_updates=len(updates)-5136,raw_training_heads=len(heads),
        read_only_continuous_metadata_finalize_sha256=sha(ROOT/'evidence/CONTINUOUS8_READONLY_FINALIZE.json'),closed_BROAD_rng_sha256=sha(control_path),model_queries_by_audit=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0274_ACTUAL5120_MATCHED_BROAD_TRAINING',len(updates),len(heads),flush=True)
if __name__=='__main__':main()
