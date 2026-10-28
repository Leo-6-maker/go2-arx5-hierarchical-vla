"""Finalization after completed eight updates; no additional optimizer/model work."""
import ast,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0274');CONTROL=ROOT.with_name('PI0273')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2,sort_keys=True)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
def finalize():
    out=ROOT/'EQUIL/continuous/training';cp=out/'checkpoint-8';state=read(cp/'trainer_state.json');assert (state['global_step'],state['max_steps'])==(8,5120)
    raw=rows(ROOT/'raw_completed_updates.jsonl');assert len(raw)==8 and {r['phase'] for r in raw}=={'continuous'} and [r['unit'] for r in raw]==list(range(1,9))
    sample=rows(out/'sample_audit.jsonl');plan=rows(ROOT/'evidence/input-audit/EQUIL/sample_audit.jsonl');assert len(sample)==16 and sample==plan[:16]
    actual=rows(out/'step_rng.jsonl');control=rows(CONTROL/'BROAD/main/training/step_rng.jsonl');assert len(actual)==16
    for a,b in zip(actual,control):
        for k in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'):assert a[k]==b[k],k
    assert len(rows(out/'S4_audit.jsonl'))==16 and len(rows(out/'full_loss.jsonl'))==16
    assert read(out/'S4_parity.json')['status']=='PASS_PI0247_U_S4_PARITY'
    gradients=read(out/'S4_first_update_gradients.json');assert set(gradients)=={'1','2','3','4'} and all(v>0 for v in gradients.values())
    assert read(cp/'PI0274_READOUT_CONTRACT.json')==read(ROOT/'design/EQUIL_READOUT_R1.json')
    before={p.relative_to(cp).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in cp.rglob('*') if p.is_file()}
    needed={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json'}|set(read(cp/'model.safetensors.index.json')['weight_map'].values());assert needed<=set(before)
    target=cp/'pi0274_provenance/CPU_ENCODER_KERNEL.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==(out/'CPU_ENCODER_KERNEL.json').read_bytes()
    else:shutil.copy2(out/'CPU_ENCODER_KERNEL.json',target)
    for name,value in before.items():assert (cp/name).stat().st_size==value['bytes'] and sha(cp/name)==value['sha256']
    write(ROOT/'evidence/CONTINUOUS8_READONLY_FINALIZE.json',dict(status='PASS_COMPLETED_CONTINUOUS8_READONLY_METADATA_FINALIZATION',source_failed_job='648278.hopper-m-02',completed_updates=8,completed_micros=16,
        defect='one remaining PI0273 provenance filename after native Trainer save',added_metadata=str(target),added_sha256=sha(target),unchanged_checkpoint_files=before,
        actual_BROAD_packets_matched=True,parity_pass=True,all_four_node_gradients=True,new_model_queries=0,new_optimizer_updates=0,repeated_continuous_updates=0,canary_comparison_still_required=True))
    print('PASS_CONTINUOUS8_READONLY_FINALIZE_NO_RETRAIN',flush=True)
def seal():
    assert read(ROOT/'evidence/CONTINUOUS8_READONLY_FINALIZE.json')['status']=='PASS_COMPLETED_CONTINUOUS8_READONLY_METADATA_FINALIZATION'
    binding=read(ROOT/'evidence/RUNTIME_BINDING_R1_EQUIL.json');assert binding['training_source_sha256']==sha(ROOT/'code/train.py')
    from train import runtime,ADAPTATIONS
    constants=[]
    def visit(code):
        import types
        for v in code.co_consts:
            if isinstance(v,str):constants.append(v)
            elif isinstance(v,types.CodeType):visit(v)
    visit(runtime['run'].__code__);assert not any(s.startswith('pi0273_provenance') for s in constants)
    assert 'pi0274_provenance/CPU_ENCODER_KERNEL.json' in constants and 'EXECUTION_FILESET_R1.json' in constants
    write(ROOT/'evidence/SAVE_METADATA_R1_CPU_CHECK.json',dict(status='PASS_ALL_CHECKPOINT_PROVENANCE_PATHS_USE_PI0274',adaptations=ADAPTATIONS,old_continuous_output_reused=True,model_queries=0,optimizer_updates=0))
    original=read(ROOT/'EXECUTION_FILESET.json');paths=[Path(r['path']) for r in original['files']]
    paths.extend(ROOT/p for p in ('code/repair_save_metadata.py','code/continuation.pbs','evidence/RUNTIME_BINDING_R1_EQUIL.json','evidence/SAVE_METADATA_R1_CPU_CHECK.json','evidence/CONTINUOUS8_READONLY_FINALIZE.json','EXECUTION_FILESET.json'))
    paths.extend((ROOT/'runtime/engineering/648278').glob('*.py'));paths.extend((ROOT/'logs').glob('648278.hopper-m-02.*'))
    paths=sorted(set(paths))
    write(ROOT/'EXECUTION_FILESET_R1.json',dict(schema='PI0274_SAME_SCIENCE_SAVE_METADATA_REPAIR_R1',release='495b17078cb57fdbd47d8257d33202c57beb058e',
        supersedes_fileset_sha256=sha(ROOT/'EXECUTION_FILESET.json'),files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],scientific_contract_unchanged=True,completed_continuous8_not_repeated=True))
    print('PASS_VERSIONED_METADATA_REPAIR_RUNTIME_SEAL',len(paths),flush=True)
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('action',choices=('finalize','seal'));a=p.parse_args();finalize() if a.action=='finalize' else seal()
