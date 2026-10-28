"""Read-only actual consumption, graph, paired packets and four Trainer inventories."""
import json
import math
import ast
import sys
from collections import Counter
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0273')
sys.path.insert(0,str(ROOT.with_name('PI0267')/'code'))
import pi0264_training_audit as mature
import pi0264_archive_trainer as archive
from prepare import seed_for,STATS_SHA
read,sha=mature.read,archive.sha
rows=mature.rows


def graph(arm,path):
    s4,losses,temporal=[rows(path/name) for name in ('S4_audit.jsonl','full_loss.jsonl','temporal_components.jsonl')]
    assert len(s4)==len(losses)==len(temporal)==10240
    mature.ROOT=ROOT;mature.graph(arm,max_steps=5120,campaign='PI0273')
    for index,(r,loss,t) in enumerate(zip(s4,losses,temporal),1):
        assert r['forward_index']==loss['step']==index
        assert not r['expert_action_in_condition'] and not r['active_action_modified']
        assert r['native_epsilon']['dtype']=='torch.bfloat16' and r['stock_initial_noise']['dtype']=='torch.float32'
        assert r['stock_vl_features']['dtype']=='torch.float32'
        assert (r['valid_points'],r['valid_transitions'])==(40,39)
        assert t['arm']=='PREFIX'
        assert r['physical_position_loss']==t['weighted_position'] and r['physical_delta_loss']==t['weighted_increment']
        assert loss['rotation']==t['weighted_rotation']
        assert all(math.isfinite(loss[k]) for k in ('total','rotation','speed_margin'))
        f=mature.f32
        base=f(f(r['native_loss']+f(f(.4)*r['physical_position_loss']))+f(f(.1)*r['physical_delta_loss']))
        extra=f(f(f(.1)*loss['rotation'])+f(f(.1)*loss['speed_margin']))
        assert base==r['total_loss'] and f(base+extra)==loss['total']
    gradients=read(path/'S4_first_update_gradients.json')
    assert set(gradients)=={'1','2','3','4'}
    assert all(math.isfinite(v) and v>0 for v in gradients.values())
    norms=rows(path/'update_gradient_norms.jsonl')
    assert len(norms)==5120 and [r['unit'] for r in norms]==list(range(1,5121))
    assert all(math.isfinite(r['accumulated_norm']) and r['microbatches']==2 and r['clip_max_norm']==1. for r in norms)
    parity=read(path/'S4_parity.json')
    assert parity['status']=='PASS_PI0247_U_S4_PARITY'
    return s4


def main():
    assert read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json')['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY'
    cpu=read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')
    assert cpu['status']=='PASS_PI0273_FULL_BASE_ENCODING_AND_BOTH_10240_SCHEDULES' and cpu['kernel']=='Prescott'
    runtime=ROOT/'code/train.py';seal=read(ROOT/'EXECUTION_FILESET.json')
    assert sha(runtime)==next(r['sha256'] for r in seal['files'] if r['path']==str(runtime))
    guard=ast.dump(ast.parse("assert os.environ.get('OPENBLAS_CORETYPE')=='Prescott'").body[0].test)
    assert any(ast.dump(n.test)==guard for n in ast.walk(ast.parse(runtime.read_text())) if isinstance(n,ast.Assert))
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==STATS_SHA
    checkpoints=[];packets={};logs={};exposures={};consumed={}
    for arm in ('BROAD','TARGET'):
        assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0273_REAL_EXACT_RESUME_CANARY'
        path=ROOT/arm/'main/training';samples=rows(path/'sample_audit.jsonl');rng=rows(path/'step_rng.jsonl');s4=graph(arm,path)
        assert samples==rows(ROOT/'evidence/input-audit'/arm/'sample_audit.jsonl') and len(samples)==len(rng)==10240
        for i,(a,b,c) in enumerate(zip(samples,rng,s4)):
            assert (a['unit'],a['micro'])==(i//2+1,i%2)
            assert all(a[k]==b[k] for k in ('unit','micro','command','condition_sha256'))
            assert b['seed']==seed_for(a['unit']) and b['mode_restored'] and math.isfinite(b['loss'])
            assert all(b[k]==c[k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
            if i%2:assert all(b[k]==rng[i-1][k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
        packets[arm]=rng;consumed[arm]=Counter((r['condition_sha256'],r['labels_masks_sha256'],r['stream']) for r in samples)
        base=[r for r in samples if r['stream']=='BASE'];supp=[r for r in samples if r['stream']=='SUPPLEMENT']
        assert [r['source_index'] for r in base]==list(range(8192)) and len(supp)==2048
        if arm=='TARGET':assert Counter(Counter(r['source_index'] for r in supp).values())=={7:272,6:24}
        for phase in ('continuous','split','main'):
            kernel=read(ROOT/arm/phase/'training/CPU_ENCODER_KERNEL.json')
            assert kernel['core']=='Prescott' and kernel['library_sha256']==cpu['kernel_sha256']
        exposures[arm]=[dict(task=k[0],command=k[1],bucket=k[2],microbatches=v) for k,v in sorted(Counter((r['task'],r['command'],r['bucket']) for r in samples).items())]
        logs[arm]={p.name:sha(p) for p in path.glob('*.jsonl')}
        for step in (2560,5120):
            cp=path/f'checkpoint-{step}';state=read(cp/'trainer_state.json')
            assert (state['global_step'],state['max_steps'])==(step,5120)
            for name in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'):assert (cp/name).is_file()
            assert read(cp/'statistics.json')==read(stats) and sha(cp/'effective_statistics.json')==STATS_SHA
            assert read(cp/'PI0273_READOUT_CONTRACT.json')==read(ROOT/'design'/f'{arm}_READOUT_R1.json')
            for name in ('EXECUTION_FILESET.json','PARAMETER_SCOPE.json','DONOR_PARAMETER_SCOPE.json','design/FITTING_CARD.json','STORAGE_ADMISSION.json','evidence/PRECISION_CONTRACT_RESOLUTION.json'):
                assert sha(cp/'pi0273_provenance'/name)==sha(ROOT/name)
            assert sha(cp/'pi0273_provenance/CPU_ENCODER_KERNEL.json')==sha(path/'CPU_ENCODER_KERNEL.json')
            entries=[]
            for p in sorted(cp.rglob('*')):
                assert not p.is_symlink()
                if p.is_file():entries.append(dict(path=p.relative_to(cp).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
            checkpoints.append(dict(arm=arm,step=step,path=str(cp),entries=entries,complete_trainer=True))
    for a,b in zip(packets['BROAD'],packets['TARGET']):
        assert all(a[k]==b[k] for k in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'))
    updates=rows(ROOT/'raw_completed_updates.jsonl');heads=rows(ROOT/'raw_training_heads.jsonl')
    for records,cap in ((updates,10528),(heads,196608)):
        assert [r['ordinal'] for r in records]==list(range(1,len(records)+1)) and len(records)<=cap
        assert all(r['arm'] in ('BROAD','TARGET') and r['phase'] in ('continuous','split','main') for r in records)
    counts=Counter((r['arm'],r['phase']) for r in updates)
    assert all(counts[a,'main']==5120 and counts[a,'continuous']>=8 and counts[a,'split']>=8 for a in ('BROAD','TARGET'))
    assert 0<=len(updates)-10272<=256
    early=read(ROOT/'design/NEW_FIT192.json')['rows'];consumption=[]
    for row in early:
        actual={}
        for arm in ('BROAD','TARGET'):
            key=(row['condition_sha256'],row['labels_masks_sha256'])
            assert consumed[arm][(*key,'BASE')]>0
            actual[arm]=sum(consumed[arm][(*key,stream)] for stream in ('BASE','SUPPLEMENT'))
        consumption.append(dict(input_id=row['input_id'],condition_sha256=row['condition_sha256'],labels_masks_sha256=row['labels_masks_sha256'],actual_seen=True,actual_counts=actual))
    archive.save(ROOT/'evidence/MAIN_TRAINING_AUDIT.json',dict(status='PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING',
        complete_checkpoints=checkpoints,NEW_FIT192_actual_consumption=consumption,exposure=exposures,actual_random_packets_matched=True,actual_inputs_match_CPU_plan=True,
        actual_generated_nodes=dict(BROAD=4,TARGET=4),actual_loss_gradient_mode_checked=True,logs=logs,
        encoder_binding=dict(kernel=cpu['kernel'],CPU_admission_library_sha256=cpu['kernel_sha256'],
            CPU_admission_sha256=sha(ROOT/'evidence/HOPPER_CPU_ADMISSION.json'),sealed_training_enforces_OPENBLAS_CORETYPE=True,
            actual_all_processed_inputs_equal_CPU_admission=True,separate_per_phase_kernel_receipts='ACTUAL_RECORDED'),
        raw_completed_updates=len(updates),technical_completed_updates=len(updates)-10272,raw_training_heads=len(heads),
        raw_heads_by_arm=dict(Counter(r['arm'] for r in heads)),code_sha256=sha(Path(__file__)),model_queries_by_audit=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING',len(checkpoints),len(updates),len(heads),flush=True)


if __name__=='__main__':main()
