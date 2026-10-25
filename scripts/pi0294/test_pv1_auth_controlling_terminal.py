"""Real local terminal mechanics with a noncredential fixture; no site login."""
import hashlib,os,sys
from pathlib import Path
import pytest
from scripts.pi0294.pv1.auth_lifecycle import masked_login


@pytest.mark.skipif(not hasattr(os,'forkpty'),reason='POSIX controlling-terminal primitive absent')
def test_detached_vendor_style_dev_tty_and_masked_stdin():
    secret='noncredential-terminal-fixture';digest=hashlib.sha256(secret.encode()).hexdigest()
    # The argv contains only a digest, never the fixture input. The child must
    # have a real controlling terminal, not merely stdin.isatty() == True.
    code=('import os,hashlib;fd=os.open("/dev/tty",os.O_RDWR);'
          'os.write(fd,b"Password: ");v=os.read(fd,128);os.close(fd);'
          'raise SystemExit(0 if hashlib.sha256(v.strip()).hexdigest()==' + repr(digest)+' else 2)')
    result=masked_login([sys.executable,'-c',code],secret,timeout_s=3,
                        process_cpu_limit_s=2,controlling_terminal=True)
    assert result['status']=='CLI_LOGIN_EXIT_SUCCESS_REQUIRES_FRESH_NATIVE_QUERY'
    assert result['password_prompt_answered'] and not result['credential_saved'] and not result['raw_output_logged']


def test_terminal_mode_never_accepts_password_in_arguments_or_nonboolean():
    with pytest.raises(ValueError):masked_login(['/bin/echo','fixture'], 'fixture',controlling_terminal=True)
    with pytest.raises(ValueError):masked_login(['/bin/true'], 'fixture',controlling_terminal=1)


def test_prompt_before_input_flush_does_not_drop_credential():
    secret='noncredential-prompt-fixture';digest=hashlib.sha256(secret.encode()).hexdigest()
    code=('import os,time,termios,hashlib;fd=os.open("/dev/tty",os.O_RDWR);'
          'os.write(fd,b"Password: ");time.sleep(.01);termios.tcflush(fd,termios.TCIFLUSH);'
          'v=os.read(fd,128);os.close(fd);raise SystemExit(0 if hashlib.sha256(v.strip()).hexdigest()=='
          +repr(digest)+' else 2)')
    result=masked_login([sys.executable,'-c',code],secret,timeout_s=3,
                        process_cpu_limit_s=2,controlling_terminal=True)
    assert result['password_prompt_answered'] and not result['raw_output_logged']
