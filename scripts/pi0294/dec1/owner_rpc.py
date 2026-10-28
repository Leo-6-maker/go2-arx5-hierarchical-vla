"""Bounded SSH subprocess adapter for a future registered cold owner.

SSH argv and native roots are supplied by the canonical private host binding.
No connection is made on import. There is no retry and no credential logging.
"""
import json
import subprocess


def invoke(argv, program, *, timeout_s, runner=subprocess.run):
    if (not isinstance(argv, (tuple, list)) or not argv or argv[0] != 'ssh'
            or type(timeout_s) not in (int,float) or not 0 < timeout_s <= 40
            or not isinstance(program, str)):
        raise ValueError('BOUND_CANONICAL_SSH_ARGV_AND_CONTROL_DEADLINE')
    result = runner(list(argv), input=program, capture_output=True, text=True, timeout=timeout_s)
    if result.returncode == 255:
        raise ConnectionError('SSH_TRANSPORT_FAILED')
    if result.returncode:
        # Native authority/identity/scheduler errors are not generic read retries.
        raise RuntimeError('NATIVE_CONTROL_PROGRAM_FAILED:'+str(result.returncode))
    return json.loads(result.stdout)


def program(module, function, values, code_root):
    """Exact published module from this new run's source capsule only."""
    if module != 'scripts.pi0294.dec1.owner_heartbeat_publish' or function not in ('publish','readback'):
        raise ValueError('EXACT_COLD_HEARTBEAT_ENTRY')
    return ('import sys,json\nsys.path[:0]='+repr([str(code_root),str(code_root)+'/src'])+'\n'
            'from '+module+' import '+function+'\n'
            'print(json.dumps('+function+'(**'+repr(values)+')))\n')
