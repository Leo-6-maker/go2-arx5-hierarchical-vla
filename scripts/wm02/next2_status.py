"""Read actual NEXT2 identities, terminal leases, gates and costs; never dispatch."""
def inspect(binding,*,gpu_query=False):
    import json,time,subprocess,os
    from pathlib import Path
    from go2arx5_vla.world_model.storage import digest
    binding=Path(binding);b=json.loads(binding.read_text());c=Path(b['campaign_root']);w=binding.parent.parent;rows=[];units={};pins={}
    for root in sorted((c/'runs').glob('next2-*')):
        f=root/'EXECUTION_SPEC.json'
        if not f.exists():continue
        s=json.loads(f.read_text())
        if s.get('next2_binding')!=str(binding):continue
        profile=s['profile'];lease=json.loads(Path(s['lease']).read_text());row=dict(run=root.name,profile=profile,source_sha=s['source_sha'],spec_sha256=digest(f),lease=lease)
        for name,pin in s.get('frozen_files',{}).items():pins[name]=pin
        native=root/'native-01/RESULT.json';neural=root/'fit-01/RESULT.json';batch=root/'batch-01/RESULT.json'
        if profile=='wm02-auto1-target'and native.exists():
            r=json.loads(native.read_text());episode=json.loads(Path(s['episode_manifest']).read_text());reference=r.get('reference_qualification')or{};row.update(kind='native',episode=episode,result=r,result_sha256=digest(native),raw_task_atoms=reference.get('complete_task_atoms',0),eligible_free_WM_atoms=24 if r['status']=='NATIVE_STAGE_PASSED'and episode['purpose']in('PROSPECTIVE','HOLD_REPEAT')else 0)
            if s.get('parallel_batch_spec'):
                parent=Path(s['parallel_batch_spec']).parent;peer=json.loads((parent/'batch-01/RESULT.json').read_text())
                if peer['status']!='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED':row['eligible_free_WM_atoms']=0
            grade=root/'native-01/PRECONTACT_CONTINUITY.private.json'
            if grade.exists():row['contact_grade']={k:v for k,v in json.loads(grade.read_text()).items()if k!='samples'};row['contact_grade_sha256']=digest(grade)
            row['result']=dict(r,reference_qualification={k:v for k,v in reference.items()if k!='windows'})
        elif profile=='wm02-auto1-target-neural'and neural.exists():row.update(kind='neural',result=json.loads(neural.read_text()),result_sha256=digest(neural))
        elif profile=='wm02-auto1-target-batch'and batch.exists():row.update(kind='batch',result=json.loads(batch.read_text()),result_sha256=digest(batch))
        else:row['kind']='OUTSTANDING_OR_INCOMPLETE'
        intent=root/'LAUNCH_INTENT.private.json'
        if intent.exists():
            v=json.loads(intent.read_text());argv=v.get('argv',[]);unit=v.get('unit')or next(x.split('=',1)[1]for x in argv if x.startswith('--unit='));q=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ExecMainStatus','-p','CPUUsageNSec','-p','ActiveState','-p','SubState','-p','MemoryPeak','-p','ExecMainStartTimestampMonotonic','-p','ExecMainExitTimestampMonotonic'],capture_output=True,text=True,timeout=10)
            units[unit]=dict(run=root.name,query_exit_code=q.returncode,**dict(line.split('=',1)for line in q.stdout.splitlines()if '='in line));row['unit']=unit
        rows.append(row)
    native=[r for r in rows if r['kind']=='native'];neural=[r for r in rows if r['kind']=='neural'];unknown=[r['run']for r in rows if r['kind']=='OUTSTANDING_OR_INCOMPLETE'];changed=[p for p,pin in pins.items()if digest(p)!=pin]
    total=dict(native_attempts=sum(r['profile']=='wm02-auto1-target'for r in rows),native_PASS=sum(r['result']['status']=='NATIVE_STAGE_PASSED'for r in native),native_FAIL=sum(r['result']['status']!='NATIVE_STAGE_PASSED'for r in native),native_ticks=sum(r['result']['actual_target_integrations']for r in native),raw_task_atoms=sum(r['raw_task_atoms']for r in native),eligible_free_WM_atoms=sum(r['eligible_free_WM_atoms']for r in native),eligible_free_WM_episodes=sum(bool(r['eligible_free_WM_atoms'])for r in native),main_updates=sum(r['result']['main_updates']for r in neural),restore_probe_updates=sum(r['result']['continuation_probe_updates']for r in neural),Actor_updates=sum(r['result']['Actor_updates']for r in neural),native_full_tasks=sum(r['result']['full_tasks']for r in native),all_worker_leases_closed=all(r['lease']['status']=='CLOSED'for r in rows),all_exact_units_terminal=bool(units)and all(u.get('MainPID')=='0'for u in units.values()),actual_units_CPU_s=sum(int(u.get('CPUUsageNSec','0'))/1e9 for u in units.values()),neural_main_CUDA_s=None if any('main_cuda_elapsed_ms'not in r['result']for r in neural)else sum(r['result']['main_cuda_elapsed_ms']/1000 for r in neural),neural_run_measured_wall_s=sum(r['result']['wall_s']for r in neural),neural_entire_run_peak_allocated_bytes=max([r['result'].get('entire_run_peak',r['result'].get('main_update_peak',{})).get('peak_allocated_bytes',0)for r in neural]or[0]))
    v=dict(schema='WM02_NEXT2_ACTUAL_STATUS_V1',observed_epoch=time.time(),plan_sha=b['plan_sha'],binding_sha256=digest(binding),scope=json.loads((binding.parent/'LEASE.json').read_text()),dispatch_cutoff_epoch=b['dispatch_cutoff_epoch'],deadline_epoch=b['deadline_epoch'],totals=total,unknown_runs=unknown,frozen_files_checked=len(pins),changed_frozen_files=changed,units=units,rows=rows,new_integrations=0,new_optimizer_updates=0)
    if gpu_query:
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);v['GPU_query']=dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr)
    return v


def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-query',action='store_true');a=p.parse_args(argv);v=inspect(a.binding,gpu_query=a.gpu_query);write_json(a.output,v);print(json.dumps(v['totals']));return 0 if not v['unknown_runs']and not v['changed_frozen_files']and v['totals']['all_worker_leases_closed']and v['totals']['all_exact_units_terminal']else 2

if __name__=='__main__':raise SystemExit(main())
