"""Bounded login-control authentication; never a GPU dispatcher or credential store."""
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import termios
import time


class HumanAuthRequired(PermissionError):
    def __init__(self,message,*,diagnostic=None):
        super().__init__(message);self.diagnostic=diagnostic or {}


class SiteClock:
    """Conservative site epoch from a timed RPC, independent of PC wall-clock skew."""
    def __init__(self,monotonic=time.monotonic):
        self.monotonic=monotonic;self.anchor=None;self.observed_monotonic=None

    def observe(self,site_epoch,sent_at,received_at,*,site_processing_s=0.):
        elapsed=received_at-sent_at
        if not 0<=site_processing_s<=elapsed or elapsed>120:
            raise ValueError('AUTH_CLOCK_REQUIRES_BOUNDED_MEASURED_RPC_INTERVAL')
        self.anchor=site_epoch+elapsed-site_processing_s
        self.observed_monotonic=received_at

    def now(self):
        if self.anchor is None:raise PermissionError('AUTH_SITE_CLOCK_NOT_OBSERVED')
        return self.anchor+self.monotonic()-self.observed_monotonic


def context():
    """Fingerprint identity/configuration, never token content or token-file timestamps."""
    cli=Path('/opt/am/python/bin/amgr')
    value=dict(uid=os.getuid(),user=os.environ['USER'],host=os.uname().nodename,
        home_resolved=str(Path.home().resolve()),
        config_home=os.environ.get('XDG_CONFIG_HOME'),cache_home=os.environ.get('XDG_CACHE_HOME'),
        state_home=os.environ.get('XDG_STATE_HOME'),amgr_binary_sha256=hashlib.sha256(cli.read_bytes()).hexdigest())
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return dict(fingerprint=hashlib.sha256(raw).hexdigest(),principal=value['user'],uid=value['uid'],
                control_host=value['host'],credential_content_read=False,TTL='UNKNOWN_NOT_EXPOSED_BY_SUPPORTED_CLI')


def masked_login(argv,secret,*,timeout_s=45):
    """Use an ordinary supported password prompt; discard every byte of CLI output."""
    if (not isinstance(secret,str) or not secret or any(c in secret for c in ('\r','\n','\0'))
            or any(secret in str(a) for a in argv) or '--password' in argv):
        raise ValueError('AUTH_CREDENTIAL_ONLY_IN_MEMORY_AND_MASKED_STDIN')
    master,slave=pty.openpty();process=None;data=b'';sent=False;reason=None;codes=set();hints=set()
    try:
        attrs=termios.tcgetattr(slave);attrs[3]&=~termios.ECHO
        termios.tcsetattr(slave,termios.TCSANOW,attrs)
        process=subprocess.Popen(argv,stdin=slave,stdout=slave,stderr=slave,close_fds=True)
        os.close(slave);slave=None;end=time.monotonic()+timeout_s
        while time.monotonic()<end:
            if select.select([master],[],[],.1)[0]:
                try:chunk=os.read(master,4096)
                except OSError:chunk=b''
                if not chunk:break
                data=(data+chunk)[-8192:]
                codes.update(int(v) for v in re.findall(rb'(?i)\bcode\s*[:=]?\s*(\d{3,6})\b',data))
                for pattern,label in (
                    (rb'(?i)invalid credentials|invalid password|incorrect password|authentication failed','SERVER_REJECTED_CREDENTIALS'),
                    (rb'(?i)connection refused|connection error|timed out|timeout|name resolution','TRANSPORT_OR_SERVER_CONNECTIVITY'),
                    (rb'(?i)certificate verify|sslerror|tls','TLS_OR_CERTIFICATE_ERROR'),
                    (rb'(?i)access denied|not authorized|not permitted','SERVER_AUTHORIZATION_DENIED')):
                    if re.search(pattern,data):hints.add(label)
                if re.search(rb'(?i)\b(otp|captcha|mfa)\b|two.factor|multi.factor|verification code',data):
                    reason='HUMAN_AUTH_ONCE_REQUIRED_MFA_OR_CAPTCHA';break
                if not sent and re.search(rb'(?i)password\s*:',data):
                    os.write(master,secret.encode()+b'\n');sent=True;data=b''
            if process.poll() is not None:break
        # A PTY can report EOF a few milliseconds before the child is reaped.
        if process.poll() is None and reason is None:
            try:process.wait(timeout=min(2,max(.01,end-time.monotonic())))
            except subprocess.TimeoutExpired:reason='AUTH_LOGIN_FINITE_TIMEOUT'
        if reason:
            if process.poll() is None:process.terminate();process.wait(timeout=5)
            if reason.startswith('HUMAN_'):raise HumanAuthRequired(reason)
            raise TimeoutError(reason)
        if process.returncode:
            if process.returncode==255:raise ConnectionError('AUTH_SSH_TRANSPORT_FAILURE_REDACTED')
            diagnostic=dict(CLI_exit_status=process.returncode,password_prompt_answered=sent,
                server_codes=sorted(codes),classified_hints=sorted(hints),raw_output_logged=False)
            if 'TRANSPORT_OR_SERVER_CONNECTIVITY' in hints and 'SERVER_REJECTED_CREDENTIALS' not in hints:
                error=ConnectionError('AUTH_LOGIN_SERVER_CONNECTIVITY_FAILURE_REDACTED')
                error.diagnostic=diagnostic
                raise error
            raise HumanAuthRequired('HUMAN_AUTH_ONCE_REQUIRED_LOGIN_REJECTED_OR_UNEXPECTED_PROMPT',diagnostic=dict(
                CLI_exit_status=process.returncode,password_prompt_answered=sent,server_codes=sorted(codes),
                classified_hints=sorted(hints),raw_output_logged=False))
        return dict(status='CLI_LOGIN_EXIT_SUCCESS_REQUIRES_FRESH_NATIVE_QUERY',password_prompt_answered=sent,
                    credential_saved=False,raw_output_logged=False)
    finally:
        data=b'';secret=None
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=5)
        if slave is not None:os.close(slave)
        os.close(master)


class Maintainer:
    """The same owner calls this for Q health and R readiness; no cached R proof reuse."""
    def __init__(self,query,renew,probe_context,*,clock=time.time,health_s=60,
                 query_upper_s=20,renew_upper_s=45,release_reserve_s=30):
        self.query,self.renew,self.probe_context=query,renew,probe_context
        self.clock,self.health_s=clock,health_s
        self.query_upper_s,self.renew_upper_s,self.release_reserve_s=query_upper_s,renew_upper_s,release_reserve_s
        self.expected_context=None;self.next_check=0.;self.failures=0

    def bind(self):
        self.expected_context=self.probe_context()['fingerprint']
        return self.ensure(force=True,allow_renew=True)

    def ensure(self,*,force=False,allow_renew=True,ready_epoch=None):
        now=self.clock()
        deadline=None if ready_epoch is None else ready_epoch+175
        if deadline is not None and (now<ready_epoch or now+self.query_upper_s+self.release_reserve_s>=deadline):
            raise TimeoutError('PV1_ORIGINAL_175S_RELEASE_INSUFFICIENT_FRESH_QUERY_TIME')
        if not force and ready_epoch is None and now<self.next_check:
            return dict(status='AUTH_HEALTH_NOT_DUE',next_check_epoch=self.next_check)
        if self.expected_context is None:raise PermissionError('AUTH_CONTEXT_NOT_BOUND')
        before=self.probe_context()['fingerprint']
        if before!=self.expected_context:raise PermissionError('AUTH_HELPER_AND_RELEASE_CONTEXT_MISMATCH')
        try:
            try:proof=self.query()
            except PermissionError as exc:
                if 'AUTH_EXPIRED_15006' not in str(exc) or not allow_renew:raise
                remaining=float('inf') if deadline is None else deadline-self.clock()
                if remaining<=self.renew_upper_s+self.query_upper_s+self.release_reserve_s:
                    raise TimeoutError('PV1_NO_LATE_AUTH_RENEWAL_PAST_ORIGINAL_RELEASE_BUDGET')
                self.renew()
                if self.probe_context()['fingerprint']!=before:
                    raise PermissionError('AUTH_RENEWAL_CHANGED_PRINCIPAL_OR_CONTEXT')
                proof=self.query()
            after=self.probe_context()['fingerprint']
            if after!=before:raise PermissionError('AUTH_QUERY_CONTEXT_CHANGED')
            observed=proof['observed_epoch'];now=self.clock()
            if not 0<=now-observed<=30:raise PermissionError('AUTH_NATIVE_PROJECT_QUERY_NOT_FRESH')
            if ready_epoch is not None and (observed<ready_epoch or now+self.release_reserve_s>=deadline):
                raise TimeoutError('PV1_R_PROOF_MUST_BE_FRESH_AFTER_READY_WITH_RELEASE_TIME')
            self.failures=0;self.next_check=now+self.health_s
            return dict(status='PASS_FRESH_PROJECT_SAME_AUTH_CONTEXT',proof=proof,
                        context_fingerprint=after,observed_epoch=observed,TTL='UNKNOWN_NOT_EXPOSED')
        except BaseException:
            self.failures+=1;self.next_check=self.clock()+min(300,30*2**min(self.failures-1,3))
            raise


def existing_job_auth(manager,native_state,settle,*,ready_epoch=None,released=False):
    """Authentication cannot abandon an existing allocation's terminal accounting."""
    if native_state=='F':
        return dict(status='NATIVE_TERMINAL_SETTLED',settlement=settle())
    if native_state not in ('Q','R'):
        return dict(status='UNKNOWN_NATIVE_STATE_PRESERVE_HANDLE_NO_RELEASE')
    if released:
        return dict(status='EXISTING_RELEASE_NO_SECOND_PUBLICATION')
    if native_state=='R' and ready_epoch is None:
        return dict(status='WAIT_EXACT_PBS_READY_NO_RELEASE')
    try:
        proof=manager.ensure(force=native_state=='R',ready_epoch=ready_epoch if native_state=='R' else None)
    except (PermissionError,ConnectionError,TimeoutError) as exc:
        return dict(status='AUTH_BLOCKED_KEEP_OWNER_AND_SETTLE_TERMINAL',
                    error_type=type(exc).__name__,new_release=False)
    return dict(status='R_AUTH_READY_NOT_YET_RELEASED' if native_state=='R' else 'Q_AUTH_HEALTH',
                authentication=proof,new_release=False)
