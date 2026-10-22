"""Cold-owner SSH mux binding; no reconnect, scheduler, auth or retry action.

A caller first establishes its canonical masters through ordinary authenticated
SSH. Inject the resulting per-site socket paths into a new inactive owner.
The fixed RPC adapter retains its deadlines and write-unknown latch.
"""
from pathlib import Path
import os
import stat
import subprocess


def mux_argv(argv, socket_path):
    if (not isinstance(argv,(list,tuple)) or len(argv)<3 or argv[0]!='ssh'
            or argv[-2] not in ('hopper','vanda-simon') or argv[-1]!='/usr/bin/python3 -'):
        raise ValueError('EXACT_EXISTING_CANONICAL_SITE_CONTROL_ARGV')
    path=Path(socket_path)
    if not path.is_absolute() or any(c in str(path) for c in '\n\r\x00'):
        raise ValueError('ABSOLUTE_PRIVATE_OWNER_SOCKET')
    # Remove only an earlier mux binding, preserving host, identity, strict
    # host-key checking and all connection/liveness deadlines.
    result=[];i=0
    while i<len(argv):
        if argv[i]=='-o' and i+1<len(argv) and argv[i+1].split('=',1)[0] in ('ControlPath','ControlMaster','ProxyCommand','CanonicalizeHostname'):
            i+=2
        else:result.append(argv[i]);i+=1
    return result[:-2]+['-o','ControlMaster=no','-o','ControlPath='+str(path),
        '-o','ProxyCommand=false','-o','CanonicalizeHostname=no']+result[-2:]


def checked_master(argv, socket_path, *, runner=subprocess.run):
    """Local socket check only; a missing master cannot open a fresh session."""
    path=Path(socket_path);parent=path.parent.stat();s=path.lstat()
    if (parent.st_uid!=os.getuid() or parent.st_mode&0o077
            or not stat.S_ISSOCK(s.st_mode) or s.st_uid!=os.getuid()):
        raise PermissionError('OWN_PRIVATE_CANONICAL_CONTROL_SOCKET')
    bound=mux_argv(argv,path)
    check=runner(['ssh','-S',str(path),'-O','check',bound[-2]],capture_output=True,text=True,timeout=5)
    if check.returncode:raise ConnectionError('CANONICAL_MUX_NOT_LIVE_NO_FRESH_CONNECTION_FALLBACK')
    return bound
