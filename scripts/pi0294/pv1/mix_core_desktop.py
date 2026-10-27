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
    need(request[FIELD]==GRANT_SHA and request['authority_release']==s.TASK_SOURCE
         and request['scope_grant_sha256']==s.GRANT_SHA
         and request['GPU_uuid']==GPU and request['evaluation_candidate']in('C50','D25')
         and request['noise_bank']==294101 and request['remaining_conditions']==24
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
