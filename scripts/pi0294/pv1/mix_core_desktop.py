"""One explicit current C/D NoMachine coexistence grant; no generic sharing."""
import csv,io,os,time
from pathlib import Path
from . import mix_accounting as a,successor_cd as s

GRANT_SHA='4751e8e452c177b62ecef1a533e65dd433103e6d9f6897f075a81f19a85d78dc'
FIELD='Core_desktop_coexistence_grant_sha256'
GPU='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
PID=3822512;UID=1002;START=137621374
BOOT='3bc113d7-9a7b-42d8-b942-6ccd573b3bf5'
NAME='/usr/NX/bin/nxnode.bin'


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_EXACT_DESKTOP_'+why)


def allowed(request):
    if FIELD not in request:return False
    from .mix_core_partial_resume import envelope
    remaining=envelope(request)['remaining_conditions']
    need(request[FIELD]==GRANT_SHA and request['authority_release']==s.TASK_SOURCE
         and request['scope_grant_sha256']==s.GRANT_SHA
         and request['GPU_uuid']==GPU and request['evaluation_candidate']in('C50','D25')
         and request['noise_bank']==294101 and request['remaining_conditions']==remaining
         and request['native_platform']=='CORELAB_RTX4080'
         and request['optimizer_update_limit']==0 and request.get('optimizer_updates',0)==0
         and 'shared_GPU_grant_sha256'not in request,
         'ONLY_NEW_CD_GRANT_NOT_OLD_B24_OR_OTHER_WORK')
    return True


def capture():
    need(os.uname().nodename=='corelab-4080s'and os.getuid()==2009,
         'ACTUAL_CORE_HOST_KERNEL_READ_ONLY')
    p=Path('/proc')/str(PID);proof=dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        observed_epoch=time.time(),present=p.exists())
    if p.exists():
        st=(p/'stat').read_text().rsplit(')',1)[1].split()
        proof.update(pid=PID,uid=p.stat().st_uid,proc_start=int(st[19]),state=st[0],
                     comm=(p/'comm').read_text().strip())
    return proof


def filter_processes(process_raw,request,*,proof,now_epoch):
    need(allowed(request),'EXPLICIT_DIRECT_HUMAN_GRANT_REQUIRED')
    need(proof['boot_id']==BOOT and 0<=now_epoch-proof['observed_epoch']<=120,
         'SAME_BOOT_AND_FRESH_ACTUAL_KERNEL_READ')
    keep=[];authorized=[]
    for line in csv.reader(io.StringIO(process_raw)):
        need(len(line)==4,'FULL_NATIVE_COMPUTE_ROW')
        uuid,pid,name,memory=[x.strip()for x in line]
        try:pid=int(pid);memory=int(memory)
        except ValueError:raise PermissionError('V5_CD_EXACT_DESKTOP_NUMERIC_PROCESS_FIELDS')
        if uuid==GPU and pid==PID:
            need(proof.get('present')is True and proof.get('pid')==PID
                 and proof.get('uid')==UID and proof.get('proc_start')==START
                 and proof.get('state')not in('Z','X')and proof.get('comm')=='nxnode.bin'
                 and name==NAME and 0<=memory<=1024,'EXACT_DESKTOP_IDENTITY_NOT_ANY_FOREIGN_COMPUTE')
            authorized.append(dict(GPU_uuid=uuid,pid=pid,name=name,memory_MiB=memory))
        else:keep.append(line)
    need(len(authorized)<=1,'NO_DUPLICATE_GPU_DESKTOP_IDENTITY')
    out=io.StringIO();csv.writer(out,lineterminator='\n').writerows(keep)
    return out.getvalue(),dict(grant_sha256=GRANT_SHA,kernel_read=proof,authorized_desktop_rows=authorized)


def recorded_device(process,request):
    """Reparse an immutable initial observation, without granting current use.

    The initial kernel and GPU reads belong to the process's recorded epoch.
    Current admission still calls observe_current, which obtains new raw reads
    and applies the original 120-second freshness requirement against now.
    """
    from .evaluation_core_identity import parse_GPU_rows
    proof=process['device']['exact_desktop_coexistence']['kernel_read']
    need(process['boot_id']==proof['boot_id']
         and process['started_epoch']<=proof['observed_epoch']<=process['observed_epoch'],
         'RECORDED_KERNEL_READ_BOUND_TO_INITIAL_PROCESS_OBSERVATION')
    raw=process['native_GPU_raw']
    filtered,receipt=filter_processes(raw['processes']['stdout'],request,
        proof=proof,now_epoch=process['observed_epoch'])
    device=parse_GPU_rows(raw['devices']['stdout'],filtered,
        GPU_uuid=request['GPU_uuid'],own_pid=process['pid'])
    need(device['total_bytes']-device['used_bytes']>=30*1024**3,
         'RECORDED_PREMODEL_30GIB_FREE')
    device['exact_desktop_coexistence']=receipt
    return device


def owner_kernel_helpers(original_bytes):
    """Cold fix of the exact original helper; never rewrite a running owner."""
    import hashlib,ast
    need(hashlib.sha256(original_bytes).hexdigest()==
         '925bb0c0ae4709df25636906c353a39d32a22ddd0758634909e5a5ff60aa08e1',
         'EXACT_ORIGINAL_OWNER_HELPER_BYTES')
    text=original_bytes.decode();old="GPU_uuid=r['GPU_uuid'])"
    need(text.count(old)==1,'ONE_ORIGINAL_OWNER_DEVICE_READER')
    text=text.replace(old,"GPU_uuid=r['GPU_uuid'],own_pid=pid if list((owner.directory/'MODEL_LOAD_INTENTS').glob('*.json'))else None,desktop_request=r if 'Core_desktop_coexistence_grant_sha256'in r else None)")
    tree=ast.parse(text);calls=[x for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='parse_GPU_rows']
    need(len(calls)==1 and {'own_pid','desktop_request'}<={k.arg for k in calls[0].keywords},'ALL_OWNER_CALLS_BIND_NEW_SCOPE_AND_OWN_MODEL_PID')
    return text.encode()
