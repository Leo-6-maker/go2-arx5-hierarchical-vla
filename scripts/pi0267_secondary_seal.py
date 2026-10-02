"""Freeze exact secondary source/budget; no physical start or model query."""
import json
import shutil
import sys
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1';OLD=BASE/'PI0266'
sys.path[:0]=[str(OLD/'code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0266_common import read,write,sha,verify_sources


def main():
    verify_sources();reset=read(OLD/'evidence/FIXED_RESET_VERDICT.json');assert reset['all_safe']
    card=read(ROOT/'design/SCENE_CARD.json');fixed=read(R1/'design/SECONDARY_CARD.json');cases=[]
    for point in ('expert','donor','N4096','R4096'):
        for r in fixed['cases']:
            family=next(f for f in card['families'] if f['family_id']==r['family'])
            cases.append(dict(case_id='WBC-'+point+'-'+r['family']+'-'+r['task']+'-c'+str(r['command']),family=family,
                stage='DIAGNOSTIC',point=point,task=r['task'],command=r['command'],expert='ARCLENGTH_EXP_CORRECTION_V1' if point=='expert' else 'NATIVE_S4_POLICY',reset_seed=family['reset_seed']))
    assert len(cases)==32
    budget=dict(planned_starts=32,hard_starts=36,hard_simulated_seconds=1476.,control_dt=.02,sim_dt=.005,decimation=4,sim_substeps=1,
        hard_control_ticks=73800,hard_simulate_calls=295200,hard_integration_substeps=295200,hard_actor_calls=147600,
        max_task_s=40.,max_normal_initialization_s=1.,max_model_queries_per_episode=100,model_queries=2400,implicit_reset_steps_included=True)
    originals={'design/SCENE_CARD.json':card,'design/TEACHER_CARD.json':dict(diagnostics=cases,collection=[]),
        'design/BUDGET_CARD.json':budget,'evidence/SOURCE_IDENTITY.json':read(OLD/'evidence/SOURCE_IDENTITY.json'),
        'evidence/SOURCE_QUALIFIED_RESET_REUSE.json':dict(source=str(OLD/'evidence/FIXED_RESET_VERDICT.json'),sha256=sha(OLD/'evidence/FIXED_RESET_VERDICT.json'),
            normal_reset_implementation_reused=True,new_family_seed_safety_checked_at_actual_start=True,new_reset_episodes=0)}
    for rel,value in originals.items():
        path=R1/rel
        if path.exists():assert read(path)==value
        else:write(path,value)
    ledger=R1/'evidence/RAW_BUDGET.json'
    if not ledger.exists():write(ledger,dict(starts=0,control_ticks=0,simulate_calls=0,actor_calls=0,simulated_seconds=0.))
    files={}
    for p in (R1/'code/pi0266_control.py',ROOT/'code/pi0267_secondary.py',ROOT/'code/pi0267_geometry_worker.py',ROOT/'code/pi0267_expert.py',ROOT/'code/pi0267_metrics_v3.py',
              ROOT/'code/pi0267_infer_worker.py',ROOT/'code/pi0267_virtual_episode.py',R1/'design/SECONDARY_CARD.json'):
        files[p.relative_to(R1).as_posix() if p.is_relative_to(R1) else '../'+p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in originals:files[rel]=sha(R1/rel)
    seal=dict(schema='PI0267_FIXED_SECONDARY_EXECUTION_V1',files=files,original_control_implementation=str(OLD/'code/pi0266_control.py'),
        original_geometry_witness_sha256=sha(OLD/'code/pi0266_geometry_worker.py'),original_control_sha256=sha(OLD/'code/pi0266_control.py'),
        model_install='Only actual native S4 outputs; original raw gate and full-H40 local witness; reference-only handoff',physics_starts=0,model_queries=0)
    path=R1/'design/TEACHER_EXECUTION_SEAL.json'
    if path.exists():
        prior=read(path)
        if prior!=seal:
            assert read(R1/'evidence/RAW_BUDGET.json')['starts']==0,'NO_IN_PLACE_HEALTHY_PHYSICAL_SOURCE_MUTATION'
            old=R1/'evidence'/('PRE_PHYSICS_SECONDARY_SEAL-'+sha(path)[:12]+'.json');assert not old.exists();path.rename(old)
            write(path,seal)
    else:write(path,seal)
    print('PASS_FIXED_SECONDARY_BUDGET_AND_SOURCE_SEAL_NO_PHYSICS',flush=True)


if __name__=='__main__':main()
