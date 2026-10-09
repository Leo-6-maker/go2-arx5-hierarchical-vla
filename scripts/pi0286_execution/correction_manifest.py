"""CPU-only full readback manifest for new admitted correction .npy members."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from contract import ROOT,sha,write


def build(data_root):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    data_root=Path(data_root);bank_path=ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json';bank=json.loads(bank_path.read_text())
    assert bank['requested_branches']==32 and bank['parent_point']=='PI0285_REPAIR_A2048-2048'
    assert bank['layouts']==list(range(285108,285124)) and bank['branch_times_seconds']==[3,12]
    plan_path=ROOT/'correction/SAMPLER_BUCKET_CONTRACT_V3/CPU_PLAN_DRAFT.json';plan=json.loads(plan_path.read_text())
    training_plan=plan_path.parent/'C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl'
    assert sha(bank_path)==plan['data_bank_sha256']=='f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d'
    assert plan['status']=='PASS_ACTUAL_BANK_ORIGINAL_BUCKET_CONDITIONAL_EQUALITY_CPU_PLAN_DRAFT'
    assert sha(training_plan)==plan['plan_sha256']=='0194583c35e93000771aed598817091a6d670b9f1720d31da6a06d6b37934f0e'
    assert plan['rows']==8192 and plan['source_counts']=={'expert':4096,'corrected_suffix':4096}
    result=dict(status='CORRECTION_NOT_ADMITTED_NO_TRAINING_TRANSFER',bank_status=bank['status'],plan_status=plan['status'],
        bank_sha256=sha(bank_path),plan_audit_sha256=sha(plan_path),files=[],failed_prefix_members=0,
        original_expert_members_copied=0,normalizer_members_copied=0,model_calls=0,physics_calls=0)
    if bank['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC':
        members={};branches=[]
        for report in bank['all_requested_slot_reports']:
            if report['slot'] not in bank['valid_slots']:continue
            assert report['status']=='PASS_FRESH_SUFFIX_STRICT_CODEC'
            identity=report['rows'][0]['source_identity']
            assert all(row['source_identity']==identity and row['source']=='corrected_suffix' for row in report['rows'])
            count=660-report['takeover_tick'];branch=[]
            for relative,expected in sorted(identity.items()):
                rel=Path(relative);assert rel.suffix=='.npy' and len(rel.parts)==3 and rel.parts[1]==report['slot']
                assert rel.parts[0] in ('policy_observation','action_target')
                p=data_root/rel;assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink()
                assert sha(p)==expected,'Correction member changed after codec admission'
                value=np.load(p,mmap_mode='r',allow_pickle=False);assert len(value)==count
                if rel.stem in ('base_camera_raw','hand_camera_raw'):assert value.shape==(count,128,128,3) and value.dtype==np.uint8
                elif rel.stem in ('state_v1','action'):assert value.shape==(count,10) and np.isfinite(value).all()
                elif rel.stem=='phase':assert value.shape==(count,) and np.isin(value,range(5)).all()
                elif rel.stem=='time_s':assert np.allclose(value,np.arange(report['takeover_tick'],660)*.05,rtol=0,atol=1e-8)
                else:raise AssertionError('Undeclared correction training member')
                item=dict(path=rel.as_posix(),sha256=expected,bytes=p.stat().st_size,shape=list(value.shape),dtype=str(value.dtype),slot=report['slot'])
                assert item['path'] not in members;members[item['path']]=item;branch.append(item['path'])
            assert len(branch)==6
            branches.append(dict(slot=report['slot'],takeover_tick=report['takeover_tick'],end_tick=660,H40_windows=len(report['rows']),members=branch,
                source_result_sha256=report['result_sha256'],teacher_sha256=report['teacher_sha256'],restore=report['restore']))
        assert len(branches)==bank['valid_branches']
        files=[members[p] for p in sorted(members)]
        tree=hashlib.sha256(json.dumps([[v['path'],v['bytes'],v['sha256']] for v in files],separators=(',',':')).encode()).hexdigest()
        result.update(status='PASS_ACTUAL_CORRECTION_NUMPY_FULL_MEMBER_READBACK_NOT_TRAINING_ADMISSION',files=files,branches=branches,
            tree_sha256=tree,total_bytes=sum(v['bytes'] for v in files),frozen_processor=bank['processor'],
            plan_path=str(training_plan),
            plan_sha256=plan.get('plan_sha256'),copy_scope='EXACT_NEW_ADMITTED_CORRECTION_NPY_ONLY',optimizer_launch_allowed=False)
    output=ROOT/'correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json'
    if output.exists():assert json.loads(output.read_text())==result
    else:write(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',default=str(ROOT/'data'));a=p.parse_args()
    print(json.dumps({k:v for k,v in build(a.data_root).items() if k not in ('files','branches')}))
