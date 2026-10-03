"""Read-only AGG/control/CPU reconciliation and both complete Trainer inventories."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from collections import Counter
from pathlib import Path
from train import ROOT,CONTROL,CONTROL_SAMPLE_SHA,CONTROL_RNG_SHA,read,read_lines,seed_for
import pi0264_training_audit as mature
import pi0264_archive_trainer as archive


def main():
    arm='AGG';output=ROOT/arm/'main/training';mature.ROOT=ROOT
    assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0270_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS'
    samples=read_lines(output/'sample_audit.jsonl');rng=read_lines(output/'step_rng.jsonl');s4=read_lines(output/'S4_audit.jsonl')
    expected=read_lines(ROOT/'evidence/input-audit/AGG/sample_audit.jsonl')
    assert samples==expected and len(samples)==len(rng)==len(s4)==8192
    assert archive.sha(CONTROL/'PREFIX/main/training/sample_audit.jsonl')==CONTROL_SAMPLE_SHA
    assert archive.sha(CONTROL/'PREFIX/main/training/step_rng.jsonl')==CONTROL_RNG_SHA
    old_samples=read_lines(CONTROL/'PREFIX/main/training/sample_audit.jsonl');old_rng=read_lines(CONTROL/'PREFIX/main/training/step_rng.jsonl')
    temporal=read_lines(output/'temporal_components.jsonl');full=read_lines(output/'full_loss.jsonl')
    assert len(temporal)==len(full)==8192
    for original,components,loss in zip(s4,temporal,full):
        assert components['arm']=='PREFIX' # The unchanged temporal hook records the inherited recipe label.
        assert original['physical_position_loss']==components['weighted_position']
        assert original['physical_delta_loss']==components['weighted_increment']
        assert loss['rotation']==components['weighted_rotation']
    for i,(a,b,c,old,old_packet) in enumerate(zip(samples,rng,s4,old_samples,old_rng)):
        unit,micro=i//2+1,i%2
        assert (a['unit'],a['micro'])==(unit,micro) and c['forward_index']==i+1
        assert all(a[k]==b[k] for k in ('unit','micro','command','condition_sha256'))
        assert b['seed']==seed_for(unit) and b['mode_restored']
        for k in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'):assert b[k]==old_packet[k]
        for k in ('native_epsilon','native_tau','stock_initial_noise'):assert b[k]==c[k]
        for k in ('unit','micro','family','task','slot','command','template','language'):assert a[k]==old[k]
        if a['slot']=='COMMON_NOMINAL':
            for k in ('condition_sha256','nonlanguage_sha256','labels_masks_sha256'):assert a[k]==old[k]
        else:assert a['slot']=='RECOVERY_INTERVENTION' and a['original_index_is_provenance_only'] and a['actual_source_uid']
    fit=read(ROOT/'design/NEW_FIT192.json')['rows'];consumed=[]
    assert len(fit)==192
    for row in fit:
        m=row['source_mapping'];actual=samples[2*(m['unit']-1)+m['micro']]
        assert actual['actual_source_uid']==m['source_uid'] and actual['source_sample_sha256']==m['sample_sha256']
        assert actual['condition_sha256']==row['condition_sha256'] and actual['labels_masks_sha256']==row['labels_masks_sha256']
        assert actual['language']==row['language']
        consumed.append(dict(input_id=row['input_id'],unit=actual['unit'],micro=actual['micro'],
            condition_sha256=actual['condition_sha256'],labels_masks_sha256=actual['labels_masks_sha256'],actual_seen=True))
    mature.graph(arm,max_steps=4096,campaign='PI0270')
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';checkpoints=[]
    for step in (2048,4096):
        cp=output/f'checkpoint-{step}';state=read(cp/'trainer_state.json');assert (state['global_step'],state['max_steps'])==(step,4096)
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'):assert (cp/name).is_file()
        assert read(cp/'statistics.json')==read(stats) and archive.sha(cp/'effective_statistics.json')==archive.sha(stats)
        contract=read(cp/'PI0270_POLICY_CONTRACT.json')
        assert contract['recovery_mapping_sha256']==archive.sha(ROOT/'RECOVERY_MAPPING.json')
        assert contract['control_sample_sha256']==CONTROL_SAMPLE_SHA and contract['control_rng_sha256']==CONTROL_RNG_SHA
        entries=[]
        for p in sorted(cp.rglob('*')):
            assert not p.is_symlink()
            if p.is_file():entries.append(dict(path=p.relative_to(cp).as_posix(),bytes=p.stat().st_size,sha256=archive.sha(p)))
        checkpoints.append(dict(arm=arm,step=step,path=str(cp),entries=entries,complete_trainer=True))
    updates=read_lines(ROOT/'raw_completed_updates.jsonl');heads=read_lines(ROOT/'raw_training_heads.jsonl')
    for rows,cap in ((updates,4368),(heads,80000)):
        assert [r['ordinal'] for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=cap
    counts=Counter((r['arm'],r['phase']) for r in updates);assert counts[arm,'main']==4096
    reserve=len(updates)-4112;assert 0<=reserve<=256
    value=dict(status='PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY',optimizer_updates=4096,complete_checkpoints=checkpoints,
        actual_inputs_match_full_CPU_plan=True,all4096_COMMON_input_label_bit_identities=True,all8192_actual_random_packets_equal_control=True,
        actual_S4_loss_gradient_mode_checks=True,NEW_FIT192_actual_consumption=consumed,
        raw_completed_updates=len(updates),technical_recovery_completed_updates=reserve,raw_training_heads=len(heads),
        sample_audit_sha256=archive.sha(output/'sample_audit.jsonl'),step_rng_sha256=archive.sha(output/'step_rng.jsonl'),
        no_model_or_gradient_queries_by_audit=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    archive.save(ROOT/'evidence/MAIN_TRAINING_AUDIT.json',value)
    print(value['status'],len(checkpoints),len(updates),len(heads),flush=True)


if __name__=='__main__':main()
