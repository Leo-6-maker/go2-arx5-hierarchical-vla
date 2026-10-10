"""Independent CPU recomputation from the preserved initialization-repair arrays."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/PI0288/initialization-repair'
EPISODE='53861d09823f4dbc8690e7cf24c4522d'
def read(p):return json.loads(p.read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    raw=OUT/'evidence'/EPISODE
    rows=[json.loads(l) for l in (raw/'physics.jsonl').read_text().splitlines()]
    model=rows[0];samples=[r for r in rows if r['kind']=='sample'];stops=[r for r in rows if r['kind']=='stop']
    assert [r['step'] for r in samples]==list(range(1,43))
    assert stops==[dict(kind='stop',reason='REGISTERED_JOINT_SAFETY',steps=42)]
    initial=json.loads((raw/'initialization.jsonl').read_text().strip())
    home=read(OUT/'cpu/HOME_SHARED_RESET_CPU_AUDIT.json')
    assert home['reset_exact'] and not home['invalid_joint_count'] and not home['illegal_touching_contacts']
    assert initial['qpos'][7:19]==samples[0]['qpos'][7:19]
    assert initial['qpos'][2]==.34 and samples[0]['qpos'][2]<.34
    violations=[]
    for j in model['joints']:
        if not j['limited']:continue
        lo,hi=j['range'];dt=model['timestep'];duration=0;peak=0;warnings=[];immediate=[];sustained=[]
        for s in samples:
            q=s['qpos'][j['qposadr']];v=max(0,lo-q,q-hi);peak=max(peak,v)
            eps=1e-4 if j['type']==3 else 1e-6;limit=1e-3 if j['type']==3 else 1e-4
            duration=duration+dt if v>eps else 0
            if v>eps:warnings.append(s['sim_time_s'])
            if v>limit:immediate.append(s['sim_time_s'])
            if duration>=.020-1e-12:sustained.append(s['sim_time_s'])
        if warnings:violations.append(dict(joint=j['name'],peak_positive_violation_rad=peak,first_warning_sim_s=warnings[0],last_warning_sim_s=warnings[-1],warning_steps=len(warnings),ending_contiguous_warning_s=duration,immediate_threshold_events=immediate,sustained_stop_events=sustained))
    assert len(violations)==2 and next(x for x in violations if x['joint']=='x5_joint3')['ending_contiguous_warning_s']>=.020-1e-12
    contacts=[s for s in samples if s['contacts']];penetrations=[s for s in samples if any(c['dist']<0 for c in s['contacts'])]
    geoms={g['id']:g for g in model['geoms']}
    contact_rows=[]
    for s in contacts:
        for c in s['contacts']:
            a,b=geoms[c['geom1']],geoms[c['geom2']]
            assert (a['name']=='floor' and b['name'] in ['FL','FR','RL','RR']) or (b['name']=='floor' and a['name'] in ['FL','FR','RL','RR'])
            contact_rows.append(dict(sample_sim_s=s['sim_time_s'],contact_state_sim_s=s['contact_time_s'],geom_a=a['name'],geom_b=b['name'],distance_m=c['dist'],normal_force_N=c['force'][0]))
    write('CONTACT_AND_TIME.json',dict(initial_feet=home['feet'],initial_contacts=home['contacts'],first_margin_contact_state_s=contacts[0]['contact_time_s'],first_negative_distance_state_s=penetrations[0]['contact_time_s'],peak_penetration_m=max(-c['distance_m'] for c in contact_rows),peak_normal_force_N=max(c['normal_force_N'] for c in contact_rows),illegal_contacts=False,all_native_warning_counts_zero=all(not any(s['native_warnings']) for s in samples),base_z_initial_m=.34,base_z_final_m=samples[-1]['qpos'][2],native_runtime_sim_s=samples[-1]['sim_time_s'],physics_integrations=42,monotonic_sample_span_s=samples[-1]['monotonic_s']-samples[0]['monotonic_s'],stop_wall_s=samples[-1]['wall_s'],contacts=contact_rows))
    write('ADJUDICATION.json',dict(status='ENV_READY_INTERFACE_BLOCKED',initialization_cpu='PASS',actual_native_initialization='PASS_HOME_APPLIED',native_startup='VALID_SAFETY_FAIL',episode=EPISODE,stop_reason='x5_joint3 below raw lower bound >1e-4 rad for continuous .020s',stop_step=42,stop_sim_s=.084,violations=violations,original_result=read(raw/'RESULT.json'),shutdown_fault='exit139 follows registered SIGINT; not the initiating safety cause',original_verdict_preserved=True,old_default_violation_preserved=True,G1='NOT_RUN_UPSTREAM_STARTUP_FAILED',G2='BLOCKED_DEPLOYMENT_GRIPPER_SPEC_INDEPENDENT',G3='NOT_RUN',G4='NOT_RUN',selected='NONE',deployment='NONE',quality_retry=False))
    compiler=[json.loads(l) for l in (OUT/'cpu/XML_COMPILER_SELF_TEST.jsonl').read_text().splitlines()]
    assert len(compiler)==1 and compiler[0]['kind']=='XML_COMPILER_SELF_TEST'
    old=ROOT/'reports/PI0288/execution'
    old_files=['REPORT.md','ADJUDICATION.json','evidence/DEFAULT_MODEL_CPU_AUDIT.json','evidence/55f23c661cdc4c8aa50a8731928dc101/physics.jsonl']
    write('ORIGINAL_PRESERVATION_AND_PHASE_ERRATUM.json',dict(original_status_unchanged='ENV_READY_INTERFACE_BLOCKED',original_files=[dict(path=str(Path('reports/PI0288/execution')/n),sha256=sha(old/n)) for n in old_files],new_finding='MuJoCo3.2.7 mj_loadXML/TryCompile self-test calls mj_step on temporary default mjData before actual native startup creation',proof=['ZERO_INTEGRATION_STACK_BOUNDARY.log','XML_COMPILER_SELF_TEST.jsonl','MJB_ZERO_INTEGRATION_BOUNDARY.jsonl'],official_source='https://github.com/google-deepmind/mujoco/blob/3.2.7/src/user/user_model.cc#L3995-L4000',old_first_step_phase='XML_COMPILER_SELF_TEST_NOT_CONTROLLED_RUNTIME',old_cpu_zero_claim_scope='No explicitly requested integration after model load; XML compilation internally steps, so not total zero integration',old_limits_and_raw_violation_unchanged=True,original_default_native_mjData_still_illegal=True,compiled_model=read(OUT/'cpu/COMPILED_MODEL_PARITY.json'),cache_is_same_model_not_new_dynamics=True,safety_observer_modified=False,earlier_uninstrumented_compiler_cpu_load_steps='UNOBSERVED; not reported as zero'))
    ledger=read(OUT/'evidence/BUDGET.json')
    write('ACCOUNTING.json',dict(budget_reset=False,ledger_launches=len(ledger['episodes']),zero_integration_native_boundary_launches=3,total_launches_conservatively_charged=len(ledger['episodes'])+3,setup_slots_conservatively_used=len(ledger['episodes'])+3,technical_replacement_launch_attempts=sum(bool(x.get('technical_replacement')) for x in ledger['episodes']),native_reference_slots_used=0,new_controlled_native_runtime_episodes=1,controlled_runtime_integrations_observed=42,compiler_integrations_directly_observed=3,compiler_steps_earlier_uninstrumented_CPU_loads='UNOBSERVED',runtime_raw_self_test_steps_preserved=2,compiler_cache_steps_observed=1,physical_and_compiler_integrations_directly_observed_lower_bound=45,prelaunch_reserved_upper_bounds={k:sum(x['reservation'][k] for x in ledger['episodes']) for k in ledger['episodes'][0]['reservation']},GPU_service_s=sum(read(OUT/'evidence'/x['episode']/'RESULT.json')['service_s'] for x in ledger['episodes'] if (OUT/'evidence'/x['episode']/'RESULT.json').exists())+8.010995119111612,CPU_core_time_total='UNOBSERVED; build wall times and CPU test wall16.46s retained',leg_RL_actual_forwards='UNOBSERVED',MPC_actual_solve_attempts='UNOBSERVED',actual_dispatch='UNOBSERVED',teacher_plans=0,data_RGB_pairs=0,GR00T_forwards=0,optimizer_updates=0,hardware=0,no_all_cap_PASS_claim=True))
    write('CPU_RECEIPT.json',dict(shared_reset_and_raw_limits='PASS',actual_mjData_pre_first_step_same_pointer_and_home='PASS_NO_INTEGRATION',compiled_model_full_serialization_byte_parity='PASS',public_camera_schema_reference_suite=dict(passed=13,skipped=1,wall_s=16.46,skip_reason='private-source callback not available on PC; prior actual-source receipt reused'),independent_saved_runtime_CPU_recompute='PASS_REPRODUCES_SAFETY_FAIL',new_model_queries=0,formal_GR00T_processor_batch='NOT_RUN_NO_REAL_DATA'))
    with (OUT/'JOINT_MARGINS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(home['joints'][0]));w.writeheader();w.writerows(home['joints'])
    print(json.dumps(dict(status='ENV_READY_INTERFACE_BLOCKED',stop_step=42,stop_sim_s=.084,violations=violations,first_contact_s=contacts[0]['contact_time_s'],first_penetration_s=penetrations[0]['contact_time_s'])))
if __name__=='__main__':main()
