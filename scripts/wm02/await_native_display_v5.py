"""Wait for the same owned X display before execing the unchanged simulator.

Only display connection probes repeat. No simulator, physics step or robot
command is started until the probe succeeds; no simulator retry is made.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time

VERSION = 'SLA_OWNED_DISPLAY_READINESS_V5'


def await_display(*, timeout=30., clock=time.monotonic, sleep=time.sleep, probe=None):
    if probe is None:
        library = ctypes.CDLL('libglfw.so.3')
        library.glfwInit.restype = ctypes.c_int
        library.glfwGetError.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        def probe():
            ok = library.glfwInit()
            if ok:
                library.glfwTerminate()
                return True,None
            message = ctypes.c_char_p()
            code = library.glfwGetError(ctypes.byref(message))
            return False,dict(code=code,message=message.value.decode(errors='replace') if message.value else None)
    began = clock();attempts = 0;first = None
    while True:
        attempts += 1;ok,error = probe()
        if first is None and error is not None:first = error
        if ok:return dict(version=VERSION,passes=True,probe_attempts=attempts,first_error=first,
                          wall_s=clock()-began,simulator_starts=0,physics_integrations=0)
        if clock()-began >= timeout:
            return dict(version=VERSION,passes=False,probe_attempts=attempts,first_error=first,
                        last_error=error,wall_s=clock()-began,simulator_starts=0,physics_integrations=0)
        sleep(.05)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('ONE_FROZEN_SIMULATOR_EXEC_COMMAND_REQUIRED')
    result=await_display();result.update(DISPLAY=os.environ.get('DISPLAY'),XAUTHORITY=os.environ.get('XAUTHORITY'))
    with a.log.open('x')as f:json.dump(result,f)
    if not result['passes']:return 70
    os.execv(command[0],command)


if __name__=='__main__':raise SystemExit(main())
