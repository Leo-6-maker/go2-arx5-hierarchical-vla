"""Narrow schedule/decoder fixtures and immutable training admission."""
import argparse
import ast
from pathlib import Path
from train import ROOT,read,sha,Pairs,torch,old
from prepare import self_check,seed_for,RELEASE
from admit import write


def fixtures():
    assert not torch.cuda.is_available();torch.set_num_threads(4)
    def forbidden(*a,**kw):raise RuntimeError('PI0273_FIXTURE_NO_LEARNED_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    self_check()
    card=read(ROOT/'design/FITTING_CARD.json')
    for arm in ('BROAD','TARGET'):
        data=Pairs.__new__(Pairs);data.card=card;data.arm=arm;data.seed=27303
        for resume in (0,4,8,2560,5119):
            data.reset_seed(27303+resume);assert data.cursor==2*resume
            visit=card['streams'][arm][data.cursor];assert visit['unit']==resume+1 and visit['micro']==0
            r=data.record(visit);assert r['seed']==seed_for(resume+1)
        assert len(card['streams'][arm])==10240
        assert [v['source_index'] for v in card['streams'][arm] if v['stream']=='BASE']==list(range(8192))
    tree=ast.parse((ROOT/'code/train.py').read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert 'readout' not in imports and 'DIRECT' not in (ROOT/'code/train.py').read_text()
    from temporal import self_check as temporal
    from unclamped_fixture import check as decoder
    from pi0260_r3_physical import install
    install()
    import pi0247_u_rollout_overlay as overlay
    assert overlay.CLAMP_NORMALIZED_STATE is False
    result=dict(status='PASS_PI0273_SCHEDULE_RESTORE_AND_UNCHANGED_PHYSICAL_CPU_FIXTURES',
        complete_base_order=True,context_match=True,no_DIRECT_hook=True,
        temporal=temporal(),noncommuting_unclamped_decoder=decoder(),
        inherited_roundtrip_receipt_sha256=sha(ROOT.with_name('PI0269')/'evidence/HOPPER_PHYSICAL_FIXTURES.json'),
        training_source_sha256=sha(ROOT/'code/train.py'),model_queries=0,learned_gradients=0,optimizer_updates=0)
    write(ROOT/'evidence/CPU_FIXTURES.json',result);print(result['status'],flush=True)


def seal():
    cpu=read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')
    assert cpu['status']=='PASS_PI0273_FULL_BASE_ENCODING_AND_BOTH_10240_SCHEDULES'
    assert cpu['fitting_card_sha256']==sha(ROOT/'design/FITTING_CARD.json')
    fixture=read(ROOT/'evidence/CPU_FIXTURES.json')
    assert fixture['training_source_sha256']==sha(ROOT/'code/train.py')
    paths=list((ROOT/'code').glob('*'))+list((ROOT/'authority').glob('*'))+list((ROOT/'design').glob('*.json'))
    for arm in ('BROAD','TARGET'):
        p=ROOT/'evidence'/f'RUNTIME_BINDING_{arm}.json';r=read(p)
        assert r['status']=='PASS_PI0273_ACTUAL_FROZEN_HELPERS_AND_HOOKS' and r['training_source_sha256']==sha(ROOT/'code/train.py')
        paths.append(p);paths.extend(Path(m['path']) for m in r['modules'])
        p=ROOT/'evidence/input-audit'/arm/'sample_audit.jsonl';assert sha(p)==cpu['sample_audits'][arm];paths.append(p)
    for row in read(ROOT.with_name('PI0270')/'EXECUTION_FILESET.json')['files']:
        p=Path(row['path'])
        if p.suffix=='.py':assert sha(p)==row['sha256'];paths.append(p)
    # Exact historical data maps and small source receipts; no repeat historical payload downloads.
    paths.extend(ROOT.with_name('PI0270')/p for p in ('code/train.py','PAIRING_PLAN.json','RECOVERY_MAPPING.json'))
    paths.extend(ROOT/p for p in ('DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json',
        'contracts/ROT/source/dataset/statistics.json','evidence/CPU_FIXTURES.json','evidence/HOPPER_CPU_ADMISSION.json',
        'evidence/BASE_ENCODING8192.json','evidence/PRECISION_CONTRACT_RESOLUTION.json','evidence/A0_SOURCE_AND_PLAN.json'))
    paths=sorted({p for p in paths if p.is_file()})
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0273_IMMUTABLE_S4_TRAINING_FILESET_V1',release=RELEASE,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
        real_canaries_still_required=True,model_queries_by_seal=0))
    print('PI0273_RUNTIME_SEALED',len(paths),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fixtures','seal'));a=p.parse_args()
    fixtures() if a.action=='fixtures' else seal()
