"""Typed scheduler-read failure for a future cold owner, never a queue success.

No retries or connections on import. Only the original bounded read retry
policy may consume QueueReadFailure. Mutation failures remain unknown writes.
"""
import ast
import json
import subprocess
import sys

EXIT=75
SCHEMA='DEC1_NATIVE_SCHEDULER_READ_FAILURE_V1'
COMMANDS={'/opt/pbs/bin/qselect','/opt/pbs/bin/qstat',
 '/cm/shared/apps/pbspro/current/bin/qselect','/cm/shared/apps/pbspro/current/bin/qstat'}

class QueueReadFailure(ConnectionError):pass

def queue_run(args,*,capture_output,text,timeout,check,runner=subprocess.run):
    if (type(args) is not list or not args or args[0] not in COMMANDS
            or not all(type(x) is str and x for x in args) or capture_output is not True
            or text is not True or check is not True or type(timeout) not in (int,float)
            or not 0<timeout<=25):
        raise ValueError('EXACT_EXISTING_BOUNDED_READ_ONLY_QUEUE_COMMAND')
    r=runner(args,capture_output=True,text=True,timeout=timeout,check=False)
    if r.returncode:
        error=dict(schema=SCHEMA,command=args,returncode=r.returncode,
                   stdout=r.stdout[-3000:],stderr=r.stderr[-3000:],queue_known=False)
        print(json.dumps(error),file=sys.stderr,flush=True)
        raise SystemExit(EXIT)
    return r

def wrap_program(original,helper_code):
    """Replace only three run calls in the exact mature read snapshot AST.

    The caller pins both byte strings in its new cold capsule. No monkeypatch
    of subprocess or in-flight process is applied, and heartbeat code rejects.
    """
    tree=ast.parse(original);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='subprocess' and n.func.attr=='run']
    if len(calls)!=3 or "with_name('qselect')" not in original or 'READ_ONLY_QUEUE_GUARD' in original:
        raise ValueError('EXACT_ORIGINAL_FULL_QUEUE_READ_PROGRAM_ONLY')
    # Preserve every original identity, exact IDs, ownership, ledger and result
    # check. Source replacement is limited to the validated three AST calls.
    if original.count('subprocess.run(')!=3:raise ValueError('EXACT_QUEUE_CALL_SPELLING')
    return '# READ_ONLY_QUEUE_GUARD\n'+helper_code+'\n'+original.replace('subprocess.run(','queue_run(')

def decode_read_result(result,*,effect):
    """Use only after bounded transport and private diagnostics record result."""
    if result.returncode==255:raise ConnectionError('SSH_TRANSPORT_FAILED')
    if result.returncode==EXIT and effect=='READ_ONLY' and not result.stdout.strip():
        try:v=json.loads(result.stderr)
        except (ValueError,TypeError):v=None
        if (type(v) is dict and set(v)=={'schema','command','returncode','stdout','stderr','queue_known'}
                and v['schema']==SCHEMA and v['queue_known'] is False
                and type(v['returncode']) is int and v['returncode']!=0
                and type(v['command']) is list and v['command'] and v['command'][0] in COMMANDS
                and all(type(x) is str for x in v['command'])
                and type(v['stdout']) is str and type(v['stderr']) is str):
            raise QueueReadFailure('NATIVE_SCHEDULER_READ_FAILED_NOT_EMPTY_QUEUE')
    if result.returncode:raise RuntimeError('NATIVE_CONTROL_PROGRAM_FAILED:'+str(result.returncode))
    return json.loads(result.stdout)
