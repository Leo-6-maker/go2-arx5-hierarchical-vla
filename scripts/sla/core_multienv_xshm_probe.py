"""Run two authenticated Xvfb MIT-SHM clients without entering a physics world."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


def segments():
    lines = Path('/proc/sysvipc/shm').read_text().splitlines()
    names = lines[0].split()
    rows = [dict(zip(names, row.split())) for row in lines[1:]]
    return {int(row['shmid']): row for row in rows}


def read_message(process):
    with selectors.DefaultSelector() as ready:
        ready.register(process.stdout, selectors.EVENT_READ)
        if not ready.select(timeout=4):
            raise TimeoutError('Xvfb SHM fixture did not respond')
        return process.stdout.readline().split()


def actor(stack, root, wrapper, check, start_display, marker):
    tmp = root/f'world-{marker}'
    tmp.mkdir(mode=0o700)
    errors = stack.enter_context((tmp/'stderr.log').open('xb'))
    environment = dict(os.environ, TMPDIR=str(tmp), LIBGL_ALWAYS_SOFTWARE='1',
                       MUJOCO_GL='egl', PATH=f'{wrapper.parent}:{os.environ["PATH"]}')
    process = subprocess.Popen([
        str(wrapper), '--auto-servernum', '--server-num', str(start_display),
        '--server-args', '-screen 0 1280x1024x24 -nolisten tcp', str(check), str(marker)
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
        text=True, bufsize=1, env=environment, start_new_session=True)

    def clean():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        process.stdin.close()
        process.stdout.close()

    stack.callback(clean)
    role, pid, shmid, display = read_message(process)
    row = segments()[int(shmid)]
    cpid, lpid = (int(row[name]) for name in ('cpid', 'lpid'))
    if (role != 'READY' or row['key'] != '0' or row['uid'] != str(os.getuid())
            or cpid != int(pid) or row['nattch'] != '2' or cpid == lpid
            or Path(os.readlink(f'/proc/{lpid}/exe')).resolve() != wrapper.with_name('Xvfb').resolve()):
        raise PermissionError('SHM creator or Xvfb attachment differs from the fixture')
    return process, dict(marker=marker, display=display, creator=cpid, xvfb=lpid,
                         shmid=int(shmid), key=0, attachments=2)


def exchange(actors, sequence):
    for process, _ in actors:
        process.stdin.write(f'{sequence}\n')
        process.stdin.flush()
    for process, identity in actors:
        row = read_message(process)
        if row != ['ACK', str(sequence), f'{(identity["marker"] << 16) | sequence:x}']:
            raise ValueError('The receiving display did not read back its own pixels')


def probe(root, wrapper, fixture):
    if (os.uname().nodename != 'corelab-4080s' or os.getuid() != 2009
            or root.stat().st_uid != os.getuid() or not root.is_dir()):
        raise PermissionError('Use the dedicated Core project scratch owner')
    binary = root/'xshm-check'
    subprocess.run(['cc', '-Wall', '-Wextra', '-Werror', '-O2', str(fixture), '-o',
                    str(binary), '-lXext', '-lX11'], check=True, timeout=15)
    start = time.monotonic()
    with ExitStack() as stack:
        first = actor(stack, root, wrapper, binary, 187, 31)
        second = actor(stack, root, wrapper, binary, 197, 47)
        actors = (first, second)
        for key in ('display', 'shmid', 'xvfb'):
            if first[1][key] == second[1][key]:
                raise RuntimeError('Fixture worlds shared their X display or segment')
        for step in range(8):
            exchange(actors, step)
        first[0].stdin.close()
        if first[0].wait(timeout=5) != 0 or first[1]['shmid'] in segments():
            raise RuntimeError('First display did not release its private segment')
        for step in range(8, 16):
            exchange((second,), step)
        second[0].stdin.close()
        if second[0].wait(timeout=5) != 0 or second[1]['shmid'] in segments():
            raise RuntimeError('Second display did not release its private segment')
    return dict(status='XSHM_PRIVATE_PAIR_CONTINUED', worlds=[first[1], second[1]],
                paired_readbacks=16, peer_readbacks_after_first_exit=8,
                seconds=time.monotonic()-start, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                xvfb_sha256=hashlib.sha256(wrapper.with_name('Xvfb').read_bytes()).hexdigest(),
                native_requests=0, native_integrations=0, model_generations=0, optimizer_updates=0)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('wrapper', type=Path)
    cli.add_argument('fixture', type=Path)
    args = cli.parse_args()
    result = probe(args.root.resolve(strict=True), args.wrapper.resolve(strict=True),
                   args.fixture.resolve(strict=True))
    with (args.root/'RESULT.private.json').open('x') as out:
        json.dump(result, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(result, sort_keys=True))
