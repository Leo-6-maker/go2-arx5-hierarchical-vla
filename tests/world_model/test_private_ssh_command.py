"""Exercise both real OpenSSH argv-joining boundaries without network access."""
import shlex
import pytest
from scripts.pi0294.ssh_command import via,OPTIONS


@pytest.mark.parametrize('program',[
    "R='/private/run-01'\nfrom pathlib import Path\nprint(Path(R))",
    'print("literal $HOME $(not-a-command) `not-a-command` and \'quoted\'")',
    'print("真实文本\\nsecond line")\nprint("a b;c")',
])
def test_program_survives_both_remote_shells_and_keeps_stdin_available(program):
    expected=['/fixture/python','-B','-c',program]
    args=via('fixture-relay','fixture-compute',expected)
    assert args[:1+len(OPTIONS)]==['ssh',*OPTIONS]
    assert args[1+len(OPTIONS)]=='fixture-relay'
    relay_shell=shlex.split(args[-1])
    assert relay_shell[:1+len(OPTIONS)]==['ssh',*OPTIONS]
    assert relay_shell[1+len(OPTIONS)]=='fixture-compute'
    joined_remote_command=' '.join(relay_shell[2+len(OPTIONS):])
    assert shlex.split(joined_remote_command)==expected
    assert all(x not in args for x in ['-n','-f'])
