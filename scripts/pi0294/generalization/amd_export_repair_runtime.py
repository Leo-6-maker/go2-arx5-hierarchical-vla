"""Registered one-time CPU git repair, reusing the original exporter/R0.

The previous failed raw namespace is a closed registered input, never a new
copy or an old worker restart. Both roots share one previously paid aggregate
output envelope plus the declared engineering metadata.
"""
import argparse,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from . import amd_export_repair_accounting as accounting,amd_adoption as a
from . import amd_export_runtime as original

ROLE=accounting.VERSION
PROFILE=accounting.PROFILE
RUN='gen1-vanda-official-export-git-repair-02'
GIT='/cm/shared/apps/git/2.40.0/bin/git'
GIT_SHA='57e12e63c8393282bf62d972edf41c748d4168bbdde97d60aa43e7bcc7d2dfca'
MODULE='/cm/shared/modulefiles/git/2.40.0'
MODULE_SHA='5892547338ed2bf25eb7adfc74d812d5b7484d5ddabc01d31bafd682815750e0'
native_allocation=original.native_allocation

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return original.sha(p)

def tools(*,run_git=False):
    # Only PATH is changed by the actual cluster module. Preserve the parent
    # Python/torch loader environment and use this exact shared installation.
    a.need(Path(GIT).is_file()and sha(GIT)==GIT_SHA
        and Path(MODULE).is_file()and sha(MODULE)==MODULE_SHA,
        'GIT_REPAIR_ACTUAL_CLUSTER_TOOL_AND_MODULE_IDENTITIES')
    before=os.environ.get('LD_LIBRARY_PATH');os.environ['PATH']=str(Path(GIT).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    a.need(os.environ.get('LD_LIBRARY_PATH')==before,'GIT_REPAIR_NO_PARENT_LOADER_CHANGE')
    if run_git:
        v=subprocess.run([GIT,'--version'],capture_output=True,text=True,timeout=5)
        a.need(v.returncode==0 and v.stderr==''and v.stdout=='git version 2.40.0\n',
            'GIT_REPAIR_REAL_COMPUTE_TOOL_ABI')
    return dict(git_sha256=GIT_SHA,module_sha256=MODULE_SHA,parent_LD_unchanged=True,
        actual_git_executed=run_git)

def aggregate_output(root,old_root):
    total=0;per_root={}
    for r in [Path(root),Path(old_root)]:
        a.need(r.is_absolute()and not r.is_symlink(),'GIT_REPAIR_REAL_REGISTERED_OUTPUT_ROOT')
        count=0
        for p in r.rglob('*'):
            a.need(not p.is_symlink(),'GIT_REPAIR_NO_HIDDEN_OUTPUT_LINK:'+str(p))
            if p.is_file():
                a.need(p.stat().st_uid==os.getuid(),'GIT_REPAIR_OWN_OUTPUT_FILE')
                count+=p.stat().st_size
        per_root[r.name]=count;total+=count
    a.need(total<=accounting.LIMITS['shared_previous_paid_output_upper_bytes']+accounting.METADATA,
        'GIT_REPAIR_BOTH_RETAINED_AND_NEW_OUTPUTS_INSIDE_ONE_PAID_ENVELOPE')
    return dict(total_bytes=total,per_root=per_root,old_files_and_costs_removed=False)

def validate(root,*,compute=True):
    a.need(type(compute)is bool,'GIT_REPAIR_EXPLICIT_NATIVE_OR_COLD_VALIDATION')
    root=Path(root);a.need(root.is_absolute()and not root.is_symlink()and os.getuid()==22182,
        'GIT_REPAIR_ORIGINAL_VANDA_UID_ROOT')
    m=read(original.member(root,'RUN_MANIFEST.json'));b=read(original.member(root,'OUTPUT_BINDING.private.json'))
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');mark=a.validate(g);items=accounting.rows(g)
    a.need(len(items)==1,'GIT_REPAIR_ACTUALLY_REGISTERED_CPU_REQUEST');row=items[0]
    request=read(original.member(root,'REQUEST.json'));role=ROLE+'__'+RUN
    a.need(m['schema']=='GEN1_VANDA_CPU_EXPORT_GIT_REPAIR_RUN_MANIFEST_V1'
        and m['scope']==ROLE and m['run_uuid']==root.name==RUN
        and m['output_root']==b['run_root']==str(root)and b['uid']==m['owner_uid']==22182
        and b['role']==role and sha(root/'RUN_MANIFEST.json')==b['run_manifest_sha256']==row['run_manifest_sha256']
        and m['execution']['profile']==PROFILE and m['limits']==accounting.LIMITS
        and request['request_id']==row['request_id']and request['submission_nonce']==row['submission_nonce']
        and sha(root/'REQUEST.json')==row['request_sha256']and row['source_sha']==m['code']['vla_commit']
        and row['status']in accounting.STATES[:2],
        'GIT_REPAIR_NEW_STANDARD_MANIFEST_PROFILE_BINDING_AND_SAME_REQUEST')
    lease=read(b['lease']);a.need(lease['status']=='ACTIVE'and lease['run_uuid']==RUN
        and lease['manifest_sha256']==b['run_manifest_sha256']
        and lease['deadline_epoch']==m['authority']['deadline_epoch']==mark['deadline_epoch']
        and mark['T_CD']<=time.time()<min(mark['deadline_epoch'],m['execution']['operation_deadline_epoch']),
        'GIT_REPAIR_OWN_ACTIVE_LEASE_EXISTING_WINDOW')
    entries=read(b['registry'])['entries'];a.need(sum(x.get('role')==role and x['path']==str(root)
        and x['run_uuid']==RUN for x in entries)==1,'GIT_REPAIR_REAL_NEW_STANDARD_ROLE')
    profiles=[x for x in read(b['environment'])['profiles']if x['name']==PROFILE]
    a.need(len(profiles)==1 and profiles[0]['manifest']==str(root/'RUN_MANIFEST.json')
        and profiles[0]['entry_module']=='scripts.pi0294.generalization.amd_export_repair_runtime'
        and profiles[0]['python']==original.ORIGINAL_PYTHON and profiles[0]['PBS_required']is True
        and profiles[0]['ngpus']==0,'GIT_REPAIR_EXPLICIT_ZERO_GPU_ENVIRONMENT_PROFILE')
    old_root=Path(mark[accounting.FIELD]['old_source_root']);old_binding=read(old_root/'OUTPUT_BINDING.private.json')
    a.need(sha(old_binding['lease'])==mark[accounting.FIELD]['old_closed_CPU_lease_sha256']
        and read(old_binding['lease'])==mark[accounting.FIELD]['old_closed_CPU_lease']
        and read(old_root/'RESOURCE_LEDGER.json')['status']=='FAILED_PRESERVED',
        'GIT_REPAIR_SOURCE_IS_EXACT_CLOSED_OLD_ROLE_NOT_ACTIVE_OR_ANONYMOUS_INPUT')
    closure=read(original.member(root,'plans/FAILED_CPU_CLOSED.private.json'))
    a.need(sha(root/'plans/FAILED_CPU_CLOSED.private.json')==row['predecessor_closure_readback_sha256']
        and sha(old_root/'RESOURCE_LEDGER.json')==closure['original_counter_sha256']
        and closure['old_root']==str(old_root),'GIT_REPAIR_ALL_OLD_FAILURE_METADATA_UNCHANGED')
    if compute:
        a.need(row['status']==accounting.STATES[1]and os.environ.get('PBS_JOBID')==row['job_id'],
            'GIT_REPAIR_ACTUAL_NATIVE_BINDING')
        release=read(root/'control/CPU_RUNNING_RELEASE.json')
        a.need(release['job_id']==row['job_id']and release['request_id']==row['request_id']
            and release['request_sha256']==row['request_sha256']
            and 0<=time.time()-release['observed_epoch']<=120,'GIT_REPAIR_FRESH_ACTUAL_CPU_RELEASE_BEFORE_INPUT_CHECKS')
    plan,packet,inputs=original.validate_source_inputs(root,m,row,input_source_root=old_root)
    tools(run_git=compute);aggregate_output(root,old_root)
    if compute:
        v=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',row['job_id']],capture_output=True,text=True,timeout=25,check=True)
        a.need(not v.stderr,'GIT_REPAIR_NATIVE_READ_SUCCESS');jobs=json.loads(v.stdout)['Jobs']
        a.need(set(jobs)=={row['job_id']},'GIT_REPAIR_EXACT_FULL_NATIVE_HANDLE')
        native_allocation(jobs[row['job_id']],job_id=row['job_id'],hostname=socket.gethostname(),request=request)
    return m,plan,packet,inputs,old_root

def run(root):
    m,plan,packet,inputs,old_root=validate(root)
    return original._run_registered_export(root,m,plan,packet,inputs,
        aggregate_guard=lambda:aggregate_output(root,old_root))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);print(json.dumps(run(p.parse_args().root)))
