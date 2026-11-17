"""Read-only alarm continuity on frozen 2ms evidence. Never rescore outcomes."""
import argparse
import json
from pathlib import Path


def alarm_runs(states):
    runs=[]
    for r in states:
        if not r['candidate_alarm']:continue
        if not runs or runs[-1]['last_step']+1!=r['step']:
            runs.append(dict(first_step=r['step'],last_step=r['step'],first_time_s=r['time_s'],last_time_s=r['time_s'],samples=1))
        else:runs[-1].update(last_step=r['step'],last_time_s=r['time_s'],samples=runs[-1]['samples']+1)
    return runs


def analyze(bundle):
    result=[]
    for artifact in bundle['safety'][0]['artifacts']:
        a=artifact['value'];states=a['states'];contact=a['original_first_illegal_contact'];runs=alarm_runs(states)
        clearance=lambda r:next(p['clearance_m']for p in r['pads']if p['pad']=='task_right_pad')
        last=states[-1]
        velocities=[(clearance(states[i])-clearance(states[i-1]))/(states[i]['time_s']-states[i-1]['time_s'])for i in range(1,len(states))]
        item=dict(id=a['window']['id'],original_qualified=a['window']['original_qualified'],artifact_sha256=artifact['sha256'],
            geometry_sha256=a['geometry_sha256'],original_result_sha256=a['original_result']['sha256'],journal_sha256=a['journal']['sha256'],
            first_alarm=a['held_reference_profile']['first_alarm'],alarm_runs=runs,original_contact=contact,
            original_lead_to_contact_s=a['held_reference_profile']['lead_to_first_recorded_contact_s'],
            final_state_s=last['time_s'],final_right_pad_clearance_m=clearance(last),minimum_right_pad_clearance_m=min(map(clearance,states)),
            maximum_sampled_pad_downward_speed_m_s=max(0.,-min(velocities)),controller_selected_reference='UNKNOWN',
            native_STOP_hook_response='UNMEASURED_NOT_INSTALLED',new_native_requests=0,new_integrations=0)
        if contact:
            item['contact_packet_time_lag_s']=last['time_s']-contact['contact_time_s']
            item['last_alarm_run_lead_to_contact_s']=contact['contact_time_s']-runs[-1]['first_time_s']
        result.append(item)
    return dict(schema='V4_SAVED_S0_ALARM_RUNS_NO_RESCORE',historical_full_result_sha256=bundle['safety'][1]['result_sha256'],
        historical_window_result_sha256=bundle['safety'][0]['result_sha256'],events=result,quarantine=bundle['quarantine'],
        new_native_requests=0,new_integrations=0,independent_readmission=False,
        causal_limitation='Receiver witness is not controller-selected reference; no full WBC restoration or dynamic STOP proof',
        derivative_scope='Saved right-pad geometry finite differences, not command limits or a forward safety bound')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('output');a=p.parse_args()
    with Path(a.output).open('x')as f:json.dump(analyze(json.loads(Path(a.bundle).read_bytes())),f,indent=2,allow_nan=False);f.write('\n')
