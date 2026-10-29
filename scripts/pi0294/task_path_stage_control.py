"""Exact cold-stage recovery; no native worker launch and no mutation replay.

Large capsules are compressed on the PC and decoded on Core. AMD only forwards
the existing SSH stream. This controller does not change the published worker.
"""
import hashlib
import json
import shlex
import subprocess
import zlib

from scripts.pi0292.durable import OwnerLock
from scripts.pi0294.dec1.owner_transport import Journal


ABSENT = ('native-entry', 'NATIVE_RUN_MANIFEST.json', 'NATIVE_SCOPE_ADOPTION.json',
          'NATIVE_COLD_PREPARATION_RESULT.json',
          'control/S1-NATIVE-COLD-PREPARATION-20261029.INTENT.json',
          'control/S1-NATIVE-COLD-PREPARATION-20261029.RECEIPT.json')

READBACK_PROGRAM = r'''
import hashlib,json,os,resource,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
root=Path(P['root']);hz=os.sysconf('SC_CLK_TCK')
boot=int(next(x.split()[1]for x in Path('/proc/stat').read_text().splitlines()if x.startswith('btime ')))
pending=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit()or int(d.name)==os.getpid():continue
 try:
  if d.stat().st_uid!=os.getuid():continue
  fields=(d/'stat').read_text().rsplit(')',1)[1].split()
  args=(d/'cmdline').read_bytes().split(bytes([0]))
  start=boot+int(fields[19])/hz
  if fields[0]!='Z'and len(args)>=2 and Path(args[0].decode()).name=='python3'and args[1]==b'-'and start>=P['intent']['started_epoch']-1:
   pending.append(dict(pid=int(d.name),proc_start=fields[19],started_epoch=start))
 except FileNotFoundError:pass
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
print(json.dumps(dict(schema='S1_COLD_STAGE_EXACT_NO_EFFECT_READBACK_V1',observed_epoch=time.time(),
 intent=P['intent'],host='corelab-4080s',account_sha256=sha(root/'control/S1_PREPARATION_ACCOUNT.json'),
 registry_sha256=sha(P['registry']),absent_paths={n:not(root/n).exists()for n in P['absent']},
 matching_stage_processes=pending,read_only=True)))
'''

ABORT_INPUT_READER_PROGRAM = r'''
import hashlib,json,os,resource,signal,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
root=Path(P['root']);d=Path('/proc')/str(P['pid']);fd=os.pidfd_open(P['pid'])
try:
 fields=(d/'stat').read_text().rsplit(')',1)[1].split()
 args=(d/'cmdline').read_bytes().split(bytes([0]))
 assert d.stat().st_uid==os.getuid()and fields[19]==P['proc_start']and fields[0]!='Z'
 assert args[:2]==[b'python3',b'-']and(d/'cwd').resolve()==Path(P['cwd'])
 assert(d/'wchan').read_text().strip()in('pipe_read','anon_pipe_read','pipe_write','anon_pipe_write','wait_woken')
 assert not(d/'task'/str(P['pid'])/'children').read_text().strip()
 assert all(not(root/n).exists()for n in P['absent'])
 assert hashlib.sha256((root/'control/S1_PREPARATION_ACCOUNT.json').read_bytes()).hexdigest()==P['account_before_sha256']
 assert hashlib.sha256(Path(P['registry']).read_bytes()).hexdigest()==P['registry_before_sha256']
 signal.pidfd_send_signal(fd,signal.SIGTERM)
 print(json.dumps(dict(status='SIGTERM_SENT_TO_EXACT_ORPHANED_STAGE_TRANSPORT',pid=P['pid'],proc_start=P['proc_start'],observed_epoch=time.time(),physics=0)))
finally:os.close(fd)
'''


def compressed_call(program, payload, *, timeout=35):
    """One bounded foreground call, with no retry and no AMD decompression."""
    packet = json.dumps(dict(program=program, payload=payload), allow_nan=False).encode()
    if len(packet) > 100 * 1024**2:
        raise ValueError('BOUNDED_COLD_CAPSULE_REQUIRED')
    wire = zlib.compress(packet, 6)
    boot = ('import sys,resource,zlib,json;sys.dont_write_bytecode=True;'
            'resource.setrlimit(resource.RLIMIT_CPU,(180,180));'
            'resource.setrlimit(resource.RLIMIT_CORE,(0,0));'
            'b=sys.stdin.buffer.read(104857601);assert len(b)<=104857600;'
            'v=json.loads(zlib.decompress(b));P=v["payload"];'
            'exec(compile(v["program"],"<published-cold-stage>","exec"))')
    opts = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=8']
    inner = shlex.join(['ssh', *opts, 'gpu4080s', shlex.join(['python3', '-c', boot])])
    return subprocess.run(['ssh', *opts, 'corelab-amd-server', inner], input=wire,
                          capture_output=True, timeout=timeout)


def reconcile_no_effect(journal, proof, *, account_sha256, registry_sha256):
    """Resolve this exact mutation only after authoritative terminal readback.

    The ERROR/INTENT stay immutable. A continuation needs a new request/journal;
    neither this function nor compressed_call replays the original call.
    """
    lock = OwnerLock(journal.root / 'RPC_OWNER.lock')
    try:
        state = journal._state()
        intent = state['intent'] if state else {}
        now = journal.clock()
        if (not state or state['status'] not in ('UNKNOWN', 'PENDING')
                or intent.get('phase') != 'S1_COMPLETE_NATIVE_COLD_STAGE_ONCE'
                or intent.get('effect') != 'MUTATION'
                or proof.get('schema') != 'S1_COLD_STAGE_EXACT_NO_EFFECT_READBACK_V1'
                or proof.get('intent') != intent or proof.get('host') != 'corelab-4080s'
                or proof.get('account_sha256') != account_sha256
                or proof.get('registry_sha256') != registry_sha256
                or proof.get('absent_paths') != {n: True for n in ABSENT}
                or proof.get('matching_stage_processes') != [] or proof.get('read_only') is not True
                or not intent['started_epoch'] <= proof.get('observed_epoch', -1) <= now
                or not 0 <= now - proof['observed_epoch'] <= 120):
            raise PermissionError('EXACT_TERMINAL_ZERO_EFFECT_READBACK_REQUIRED')
        digest = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
        journal._event(intent, 'NATIVE_NO_EFFECT_RECONCILED', proof=proof, proof_sha256=digest)
        journal._save(journal.state_path, dict(status='CONFIRMED', intent=intent,
                      resolution='ACTUAL_NO_EFFECT_AFTER_TRANSPORT_TIMEOUT', proof_sha256=digest))
        return dict(status='CONFIRMED_ZERO_EFFECT_NO_REPLAY', operation_id=intent['operation_id'],
                    proof_sha256=digest, original_error_preserved=True)
    finally:
        lock.close()
