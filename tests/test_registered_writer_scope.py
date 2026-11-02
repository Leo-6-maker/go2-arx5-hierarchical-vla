"""Project custody, not a global same-user /proc descriptor scan."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
from scripts.pi0294.multitask import book_reset as reset


def fixture(tmp_path):
    root=tmp_path/'previous';root.mkdir()
    proc=tmp_path/'proc';proc.mkdir()
    scope=dict(schema='REGISTERED_PROJECT_WRITER_SCOPE_V1',root=str(root),
        unit='previous.service',InvocationID='a'*32,ControlGroup='/units/previous.service',
        host=os.uname().nodename,uid=os.getuid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        processes=[dict(PID=100,start_ticks=123)])
    def process(pid,parent,start,name='project worker',writer=False):
        p=proc/str(pid);(p/'fd').mkdir(parents=True);(p/'fdinfo').mkdir()
        fields=['S',str(parent)]+['0']*17+[str(start)]+['0']*4
        (p/'stat').write_text(str(pid)+' ('+name+') '+' '.join(fields))
        if writer:
            (p/'fd/3').symlink_to(root/'raw.bin');(p/'fdinfo/3').write_text('flags:\t0100002\n')
        return p
    process(100,1,123)
    return root,proc,scope,process


def test_live_descendant_retaining_writer_is_found_without_external_registration(tmp_path):
    root,proc,scope,process=fixture(tmp_path)
    process(101,100,124,name='worker ) unusual name',writer=True)
    assert reset.live_writers(root,scope=scope,proc=proc)==[
        dict(PID=101,fd=3,relative_path='raw.bin')]


def test_unrelated_same_uid_systemd_descriptors_are_never_opened(tmp_path,monkeypatch):
    root,proc,scope,process=fixture(tmp_path)
    daemon=process(99,1,99,name='systemd --user',writer=True)
    original=os.readlink
    def readlink(path,*args,**kwargs):
        if str(path).startswith(str(daemon)):
            raise AssertionError('UNRELATED_SYSTEM_MANAGER_MUST_NOT_BE_INSPECTED')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'readlink',readlink)
    assert reset.live_writers(root,scope=scope,proc=proc)==[]


def test_denied_related_descriptor_stays_fail_closed(tmp_path,monkeypatch):
    root,proc,scope,process=fixture(tmp_path)
    child=process(101,100,124,writer=True)
    original=os.readlink
    def readlink(path,*args,**kwargs):
        if str(path).startswith(str(child)):
            raise PermissionError('DENIED_RELEVANT_DESCRIPTOR')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'readlink',readlink)
    with pytest.raises(PermissionError,match='DENIED_RELEVANT'):
        reset.live_writers(root,scope=scope,proc=proc)


def test_pid_reuse_and_ambiguous_scope_are_rejected(tmp_path):
    root,proc,scope,_=fixture(tmp_path)
    for change in ('reuse','duplicate','boot','missing'):
        bad=deepcopy(scope)
        if change=='reuse':bad['processes'][0]['start_ticks']+=1
        elif change=='duplicate':bad['processes']*=2
        elif change=='boot':bad['boot_id']='another boot'
        else:bad=None
        with pytest.raises(PermissionError):reset.live_writers(root,scope=bad,proc=proc)


def test_denied_registered_identity_is_not_treated_as_process_exit(tmp_path,monkeypatch):
    root,proc,scope,_=fixture(tmp_path)
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path==proc/'100/stat':raise PermissionError('DENIED_REGISTERED_IDENTITY')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(PermissionError,match='DENIED_REGISTERED_IDENTITY'):
        reset.live_writers(root,scope=scope,proc=proc)


def test_process_membership_change_during_inspection_is_unknown(tmp_path,monkeypatch):
    root,proc,scope,process=fixture(tmp_path)
    process(101,100,124,writer=True)
    original=os.readlink
    def readlink(path,*args,**kwargs):
        if str(path).startswith(str(proc/'101')) and not(proc/'102').exists():
            process(102,100,125)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'readlink',readlink)
    with pytest.raises(PermissionError,match='PROCESS_SET_CHANGED'):
        reset.live_writers(root,scope=scope,proc=proc)
