"""Site execution admission for the same frozen shared recipe.

Core keeps its original node-local contract. Vanda requires its own live PBS
allocation and pinned project/storage proof; a login host cannot load a model.
"""
from datetime import datetime,timezone,timedelta
import json,os,subprocess,time,re,math
from pathlib import Path

GIB=1024**3
PROJECT_SCRATCH='/scratch/projects/CFP04/CFP04-CF-007'

def project_quota(text,*,user,report_epoch,observed_epoch):
    """Read the exact directory quota used by the site's hpc space helper."""
    def field(label):
        values=re.findall(r'^\s*'+re.escape(label)+r':\s*(.+?)\s*$',text,re.M)
        if len(values)!=1:raise PermissionError('EXACT_NATIVE_PROJECT_QUOTA_FIELD:'+label)
        return values[0]
    if (field('Path')!='/ifs/hpc'+PROJECT_SCRATCH or field('Type')!='directory'
            or field('Enforced')!='Yes' or field('Hard Exceeded')!='No'
            or field('Thresholds On')!='fslogicalsize'):
        raise PermissionError('EXACT_ENFORCED_PROJECT_DIRECTORY_QUOTA')
    def size(value,up):
        match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGT])',value)
        if not match:raise PermissionError('EXACT_NATIVE_PROJECT_QUOTA_SIZE')
        scale=1024**('KMGT'.index(match[2])+1)
        return (math.ceil if up else math.floor)((float(match[1])+(.005 if up else -.005))*scale)
    return dict(user=user,scope=PROJECT_SCRATCH,report_epoch=report_epoch,observed_epoch=observed_epoch,
        source='NATIVE_ENFORCED_PROJECT_DIRECTORY_QUOTA',
        used_bytes=size(field('FSLogical(W/O Overhead)'),True),
        hard_bytes=size(field('Hard Threshold'),False))

def quota_from_hpc_space(text,user,observed_epoch):
    """Conservative parsing of the site's own rounded per-user scratch report."""
    pattern=re.compile(r'^\s*/scratch/'+re.escape(user)+r'\s+([0-9.]+)([KMGT])\s+([0-9.]+)([KMGT])\s+[0-9.]+%\s*$',re.M)
    rows=pattern.findall(text)
    if len(rows)!=1:raise PermissionError('SHARED_EXACT_NATIVE_HPC_SCRATCH_QUOTA_ROW')
    usage,unit,quota,q_unit=rows[0];units={k:1024**i for i,k in enumerate('KMGT',1)}
    # hpc prints two decimals: round usage upward and quota downward.
    used=math.ceil((float(usage)+.005)*units[unit]);hard=math.floor((float(quota)-.005)*units[q_unit])
    return dict(user=user,report_epoch=observed_epoch,used_bytes=used,hard_bytes=hard,
        scope='/scratch/'+user,source='NATIVE_HPC_SPACE_CONSERVATIVE_ROUNDED_ROW')

def validate_vanda(context,job,*,now,uid,user,hostname,job_id):
    project=context.get('project_receipt',{});p=project.get('project',{})
    quota=context.get('quota_receipt',{})
    nodes={x.split('/')[0].split('.')[0] for x in job.get('exec_host','').split('+')}
    day=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date().isoformat()
    if (context.get('schema')!='SHARED_RAW_VANDA_A40_EXECUTION_V1'
            or context.get('job_id')!=job_id or context.get('owner_uid')!=uid
            or context.get('owner_user')!=user or hostname.split('.')[0] not in nodes
            or hostname.startswith('stdct-login')
            or job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!=user
            or job.get('project')!='CFP04-CF-007'
            or int(job.get('Resource_List',{}).get('ngpus',0))!=1):
        raise PermissionError('SHARED_VANDA_OWN_LIVE_SINGLE_GPU_ALLOCATION')
    balance=(project.get('balance_su',0)>0 or
             project.get('cpu_hrs',0)>0 and project.get('gpu_hrs',0)>0)
    submitted=context.get('submission_epoch',now)
    continued=context.get('authorization_mode')=='SAME_PBS_ALLOCATION_AFTER_FRESH_SUBMISSION'
    reference=submitted if continued else now
    if continued and (not submitted<=now or not context.get('submission_nonce')
            or job.get('Variable_List',{}).get('PV1_SUBMISSION_NONCE')!=context['submission_nonce']
            or job.get('Variable_List',{}).get('PV1_REQUEST_ID')!=context.get('science_run_id')):
        raise PermissionError('SHARED_EXACT_ONCE_SUBMITTED_PBS_ALLOCATION')
    if (p.get('name')!='CFP04-CF-007' or not p.get('active') or user not in p.get('users',[])
            or not p.get('start_date','9999')<=day<=p.get('end_date','0000')
            or not 0<=reference-project.get('observed_epoch',0)<=1800
            or not balance or any(not math.isfinite(project[k])for k in ('balance_su','cpu_hrs','gpu_hrs')if k in project)):
        raise PermissionError('SHARED_VANDA_FRESH_PROJECT_ENTITLEMENT')
    if (quota.get('user')!=user or quota.get('scope')!=PROJECT_SCRATCH
            or not 0<=now-quota.get('observed_epoch',0)<=120
            or not 0<=now-quota.get('report_epoch',0)<=7200
            or quota.get('hard_bytes',0)-quota.get('used_bytes',0)
                <context.get('predicted_new_peak_bytes',0)+20*GIB):
        raise PermissionError('SHARED_VANDA_REAL_QUOTA_HEADROOM')
    if not now<context.get('allocation_deadline_epoch',0):
        raise PermissionError('SHARED_VANDA_FINITE_ALLOCATION')
    return context

def mount_type(path):
    rows=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=left.split()[4]
        if str(path)==mount or str(path).startswith(mount.rstrip('/')+'/'):
            rows.append((len(mount),right.split()[0]))
    if not rows:raise PermissionError('SHARED_SITE_STORAGE_IDENTITY_UNKNOWN')
    return max(rows)[1]

def site_paths(root,manifest,bindings):
    from scripts.pi0294.native_dataset import sha
    site=manifest.get('execution_site','CORE')
    if site=='CORE':
        if os.uname().nodename!='corelab-4080s':
            raise PermissionError('SHARED_CORE_ORIGINAL_NODE_REQUIRED')
        from scripts.pi0294.e1.smoke import node_local
        return node_local
    if site!='VANDA_A40':raise PermissionError('SHARED_EXPLICIT_EXECUTION_SITE')
    file=root/'SITE_CONTEXT.json'
    if manifest.get('sealed_inputs',{}).get(file.name)!=sha(file):
        raise PermissionError('SHARED_PINNED_SITE_CONTEXT')
    context=json.loads(file.read_bytes());job_id=os.environ.get('PBS_JOBID')
    if not job_id:raise PermissionError('SHARED_VANDA_PBS_REQUIRED_BEFORE_MODEL_IMPORT')
    q=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',job_id],
        capture_output=True,text=True,timeout=20,check=True)
    jobs=json.loads(q.stdout)['Jobs']
    matches=[v for k,v in jobs.items()if k.split('.')[0]==job_id.split('.')[0]]
    if len(matches)!=1:raise PermissionError('SHARED_EXACT_CURRENT_PBS_JOB')
    validate_vanda(context,matches[0],now=time.time(),uid=os.getuid(),user=os.environ.get('USER'),
        hostname=os.uname().nodename,job_id=job_id)
    if (context.get('run_uuid')!=root.name
            or context.get('source_commit')!=manifest.get('code_source_sha')):
        raise PermissionError('SHARED_SITE_RUN_AND_SOURCE_IDENTITY')
    storage=context.get('storage_roots',[])
    def approved_path(value):
        path=Path(value).resolve(strict=True)
        permitted=[r for r in storage if str(r.get('root','')).startswith('/scratch/')
            and path.is_relative_to(Path(r['root']).resolve(strict=True))]
        if not permitted or not any(mount_type(path)==r['fstype'] for r in permitted):
            raise PermissionError('SHARED_VANDA_SEALED_ACTUAL_STORAGE_BINDING')
        return path
    for value in bindings.get('raw_bindings',{}).values():approved_path(value)
    return approved_path
