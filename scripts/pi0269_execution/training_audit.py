"""Reconcile actual inputs, packets, S4 audits and four complete Trainers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from collections import Counter
from pathlib import Path
from train import ROOT,read,seed_for
import pi0264_training_audit as mature
import pi0264_archive_trainer as archive

def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def main():
    mature.ROOT=ROOT;paired={};checkpoints=[];card=read(ROOT/'design/APPROVED_RUN_CARD.json')
    fit=read(ROOT/'design/FIT112_FIRST2048.json');consumption=[]
    for arm in ('UNIFORM','PREFIX'):
        assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0269_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        output=ROOT/arm/'main/training'
        expected=lines(ROOT/'evidence/input-audit/common/sample_audit.jsonl')
        expected=[dict(r,arm=arm) for r in expected]
        samples=lines(output/'sample_audit.jsonl');rng=lines(output/'step_rng.jsonl');s4=lines(output/'S4_audit.jsonl')
        assert samples==expected and len(samples)==len(rng)==len(s4)==8192
        temporal=lines(output/'temporal_components.jsonl');full=lines(output/'full_loss.jsonl')
        assert len(temporal)==len(full)==8192
        treatment='uniform' if arm=='UNIFORM' else 'weighted'
        for original,components,loss in zip(s4,temporal,full):
            assert components['arm']==arm
            assert original['physical_position_loss']==components[treatment+'_position']
            assert original['physical_delta_loss']==components[treatment+'_increment']
            assert loss['rotation']==components[treatment+'_rotation']
        seen={(r['family'],r['index'],r['language']):r for r in samples}
        for row in fit['rows']:
            actual=seen[row['family'],row['index'],row['language']]
            assert actual['condition_sha256']==row['processed']['ROT']['condition_sha256']
            consumption.append(dict(arm=arm,input_id=row['input_id'],actual_seen=True,condition_sha256=actual['condition_sha256']))
        for i,(a,b,c) in enumerate(zip(samples,rng,s4)):
            unit,micro=i//2+1,i%2
            assert (a['unit'],a['micro'])==(unit,micro)
            assert all(a[k]==b[k] for k in ('unit','micro','command','condition_sha256'))
            assert b['seed']==seed_for(unit) and b['mode_restored'] and c['forward_index']==i+1
            packet=tuple(b[k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
            assert packet==tuple(c[k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
            if micro==0:paired[arm,unit]=packet
            else:assert paired[arm,unit]==packet
        mature.graph(arm,max_steps=4096,campaign='PI0269')
        for step in (2048,4096):
            cp=output/f'checkpoint-{step}';state=read(cp/'trainer_state.json')
            assert (state['global_step'],state['max_steps'])==(step,4096)
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'):assert (cp/name).is_file()
            assert read(cp/'statistics.json')==read(ROOT/'contracts'/'ROT'/'source/dataset/statistics.json')
            assert archive.sha(cp/'effective_statistics.json')==card['statistics_sha256']
            entries=[]
            for p in sorted(cp.rglob('*')):
                assert not p.is_symlink()
                if p.is_file():entries.append(dict(path=p.relative_to(cp).as_posix(),bytes=p.stat().st_size,sha256=archive.sha(p)))
            checkpoints.append(dict(arm=arm,step=step,path=str(cp),entries=entries,complete_trainer=True))
    assert all(paired['UNIFORM',u]==paired['PREFIX',u] for u in range(1,4097))
    updates=lines(ROOT/'raw_completed_updates.jsonl');heads=lines(ROOT/'raw_training_heads.jsonl')
    for records,cap in [(updates,8480),(heads,160000)]:
        assert [r['ordinal'] for r in records]==list(range(1,len(records)+1)) and len(records)<=cap
    counts=Counter((r['arm'],r['phase']) for r in updates);assert counts['UNIFORM','main']==counts['PREFIX','main']==4096
    reserve=len(updates)-8224;assert 0<=reserve<=256
    result=dict(status='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY',optimizer_updates={'UNIFORM':4096,'PREFIX':4096},complete_checkpoints=checkpoints,
        actual_inputs_match_Hopper_CPU_plan=True,actual_random_packets_equal=True,actual_S4_gradient_loss_mode_checks=True,
        raw_completed_updates=len(updates),technical_recovery_completed_updates=reserve,raw_training_heads=len(heads),FIT112_consumption=consumption,
        completed_updates_sha256=archive.sha(ROOT/'raw_completed_updates.jsonl'),training_heads_sha256=archive.sha(ROOT/'raw_training_heads.jsonl'),
        model_queries_by_audit=0,gradient_queries_by_audit=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    archive.save(ROOT/'evidence/MAIN_TRAINING_AUDIT.json',result)
    print('PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY',len(checkpoints),len(updates),len(heads),flush=True)

if __name__=='__main__':main()
