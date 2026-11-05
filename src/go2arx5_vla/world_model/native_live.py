"""Full-body live observation over an existing authenticated command transport."""
import base64
import hashlib
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .live_source import SCHEMA, MAX_STATE, header
from .native_replay import apply_frame, forbid_integration
from .storage import digest, write_json
from scripts.pi0294.task_state import parse_state

MAX_LINE = 3 * 1024 * 1024


class LiveReceiver:
    """Keep one latest snapshot; no unbounded frame queue or control channel."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.message = None
        self.row = None
        self.last_received = None
        self.last_advanced = None
        self.advanced = False
        self.frames = 0
        self.advances = 0
        self.closed = False
        self.error = None

    def accept(self, raw, now=None):
        now = time.monotonic() if now is None else now
        if len(raw) > MAX_LINE:
            raise ValueError('OVERSIZED_OBSERVER_MESSAGE')
        value = json.loads(raw)
        expected = dict(schema=SCHEMA, episode_id=self.config['episode_id'],
                        source_sha=self.config['source_sha'], scene_sha256=self.config['scene']['sha256'],
                        native_binary_sha256=self.config['remote']['native_binary_sha256'],
                        spec_sha256=self.config['remote']['execution_spec']['sha256'])
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError('LIVE_SOURCE_IDENTITY_MISMATCH')
        if not isinstance(value['sequence'], int) or value['sequence'] < 1:
            raise ValueError('INVALID_SNAPSHOT_SEQUENCE')
        row = None
        if value['packet'] is not None:
            blob = base64.b64decode(value['packet'], validate=True)
            step, stamp = header(blob)
            if (step != value['step'] or stamp != value['sim_time_s']
                    or hashlib.sha256(blob).hexdigest() != value['packet_sha256']
                    or not math.isfinite(value['state_age_s']) or value['state_age_s'] < 0):
                raise ValueError('INVALID_BOUND_NATIVE_PACKET')
            row = parse_state(blob)
            for c in row['contacts']:
                if min(c['geom0'], c['geom1']) < 0 or max(c['geom0'], c['geom1']) >= self.config['ngeom']:
                    raise ValueError('CONTACT_SCENE_MISMATCH')
        with self.lock:
            if self.message is not None:
                if value['sequence'] <= self.message['sequence']:
                    raise ValueError('SNAPSHOT_SEQUENCE_REGRESSED')
                if self.row is not None:
                    if row is None or row['step'] < self.row['step']:
                        raise ValueError('NATIVE_TICK_REGRESSED')
                    if row['step'] == self.row['step'] and value['packet_sha256'] != self.message['packet_sha256']:
                        raise ValueError('NATIVE_STATE_CHANGED_AT_SAME_TICK')
            if row is not None and (self.row is None or row['step'] > self.row['step']):
                self.advanced = self.row is not None
                self.advances += int(self.advanced)
                self.last_advanced = now
            self.message, self.row, self.last_received = value, row, now
            self.frames += 1

    def view(self, *, now=None, stale_after=1.):
        now = time.monotonic() if now is None else now
        with self.lock:
            status = 'WAITING'
            age = None if self.last_received is None else max(0., now - self.last_received)
            if self.error:
                status = 'DATA_ERROR'
            elif self.closed:
                status = 'STREAM_ENDED'
            elif age is not None and age > stale_after:
                status = 'DISCONNECTED'
            elif self.message and self.message['lease_status'] != 'ACTIVE':
                status = 'SEALED' if self.message['lease_status'] == 'CLOSED' else 'OWNER_' + self.message['lease_status']
            elif self.row is not None:
                state_age = self.message['state_age_s'] + age
                status = 'PAUSED' if state_age > stale_after or now - self.last_advanced > stale_after else ('LIVE' if self.advanced else 'WAITING_FOR_ADVANCE')
            return dict(status=status, row=self.row, message=self.message,
                        receive_age_s=age, source_state_age_s=None if self.row is None else self.message['state_age_s'] + age,
                        frames=self.frames, native_advances=self.advances)


class ObserverTransport:
    def __init__(self, config, receiver, max_wall, fps):
        command = config['observer_command']
        if not command or not all(isinstance(a, str) and a for a in command):
            raise ValueError('EXISTING_AUTHENTICATED_COMMAND_ARGV_REQUIRED')
        # An explicit argv, never shell=True; the caller supplies its existing SSH route.
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, start_new_session=True)
        self.receiver = receiver
        self.local_stop_requested = False
        self.stderr = bytearray()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.errors = threading.Thread(target=self.read_errors, daemon=True)
        self.reader.start()
        self.errors.start()
        source = Path(__file__).with_name('live_source.py').read_text()
        bootstrap = source + '\nemit(' + repr(config['remote']) + ', max_wall=' + repr(max_wall) + ', fps=' + repr(fps) + ')\n'
        try:
            self.process.stdin.write(bootstrap.encode())
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            self.close()
            raise

    def read(self):
        try:
            while raw := self.process.stdout.readline(MAX_LINE + 1):
                self.receiver.accept(raw)
        except Exception as error:
            self.receiver.error = type(error).__name__ + ': ' + str(error)
        finally:
            self.receiver.closed = True

    def read_errors(self):
        while block := self.process.stderr.read(1024):
            self.stderr.extend(block)
            del self.stderr[:-8192]

    def close(self):
        if self.process.poll() is None:
            try:
                # Let a finite source finish despite SSH/model-load startup skew.
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.local_stop_requested = True
                self.process.terminate()  # Only our observer transport, never the simulator.
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        self.reader.join(timeout=3)
        self.errors.join(timeout=3)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


def load_scene(config):
    import mujoco
    if mujoco.__version__ != '3.2.7' or digest(config['scene']['path']) != config['scene']['sha256']:
        raise ValueError('EXACT_NATIVE_SCENE_AND_MUJOCO327_REQUIRED')
    if config['remote']['scene_sha256'] != config['scene']['sha256']:
        raise ValueError('LOCAL_REMOTE_SCENE_MISMATCH')
    if config['remote']['source_sha'] != config['source_sha'] or config['remote']['episode_id'] != config['episode_id']:
        raise ValueError('LOCAL_REMOTE_RUN_MISMATCH')
    model = mujoco.MjModel.from_binary_path(config['scene']['path'])
    if (model.nq, model.nv, model.nu, model.na, model.nplugin) != (34, 32, 20, 0, 0):
        raise ValueError('NATIVE_MODEL_ABI')
    if mujoco.get_mjcb_control() is not None:
        raise ValueError('LIVE_OBSERVER_FORBIDS_CONTROL_CALLBACKS')
    return mujoco, model, mujoco.MjData(model)


def watch(config, output, *, max_wall=300., fps=15., headless=False):
    from collections import Counter
    from scripts.pi0294.task_state import ActualState
    from PIL import Image, ImageDraw
    if not 1 <= max_wall <= 86400 or not 1 <= fps <= 30:
        raise ValueError('FINITE_VIEW_ENDPOINT_AND_RATE_REQUIRED')
    output = Path(output)
    output.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    mujoco, model, data = load_scene(config)
    config = dict(config, ngeom=int(model.ngeom))
    receiver = LiveReceiver(config)
    actual = ActualState(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [.28, 0., .45], 1.8, 135., -23.
    counts, rendered, first_step, last_step = Counter(), 0, None, None
    last_sequence = 0
    ui = handle = renderer = transport = None
    snapshots = []
    calls = []
    failure = None
    window_closed = []
    try:
        with forbid_integration(mujoco) as calls:
            if headless:
                model.vis.global_.offwidth, model.vis.global_.offheight = 960, 640
                renderer = mujoco.Renderer(model, height=640, width=960)
            else:
                import tkinter as tk
                from mujoco import viewer
                ui = tk.Tk()
                ui.title('WM02 · 实际采集场景 · 只读')
                label = tk.StringVar(value='等待采集状态…')
                tk.Label(ui, textvariable=label, justify='left', padx=12, pady=10).pack()
                ui.protocol('WM_DELETE_WINDOW', lambda: window_closed.append(True))
                handle = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
                with handle.lock():
                    handle.cam.lookat[:] = camera.lookat
                    handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            transport = ObserverTransport(config, receiver, max_wall, fps)
            while time.monotonic() - started < max_wall:
                if window_closed:
                    break
                tick = time.monotonic()
                view = receiver.view()
                counts[view['status']] += 1
                message, row = view['message'], view['row']
                text = view['status'] + ' | ' + config['episode_id']
                if row is not None:
                    apply_frame(mujoco, model, data, row)
                    state = actual.update(row)
                    first_step = row['step'] if first_step is None else first_step
                    last_step = row['step']
                    text += f"\nstep {row['step']} | sim {row['time_s']:.3f}s | source age {view['source_state_age_s']:.2f}s | receive age {view['receive_age_s']:.2f}s"
                    text += '\nTCP m ' + ', '.join(f'{v:+.4f}' for v in state['tcp'][:3, 3])
                    text += f" | native pad contacts L/R {state['left_object_contact']}/{state['right_object_contact']}"
                    text += '\nstage ' + str(message['command'].get('stage', 'unknown'))
                    if message['command'].get('installed_horizon_goal_m') is not None:
                        text += ' | installed horizon goal m ' + ', '.join(f'{v:+.4f}' for v in message['command']['installed_horizon_goal_m'])
                    if headless and message['sequence'] != last_sequence:
                        renderer.update_scene(data, camera=camera)
                        image = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, 960, 100), fill=(20, 20, 20))
                        draw.multiline_text((8, 6), text, fill='white', spacing=4)
                        image.save(output / 'latest.private.png')
                        if rendered == 0:
                            image.save(output / 'first.private.png')
                        rendered += 1
                    last_sequence = message['sequence']
                    if len(snapshots) < 3 or snapshots[-1]['native_step'] != row['step']:
                        snapshots.append(dict(native_step=row['step'], sim_time_s=row['time_s'], status=view['status']))
                        snapshots = snapshots[-100:]
                if handle is not None:
                    if not handle.is_running():
                        break
                    with handle.lock():
                        handle.perturb.active = 0
                    handle.sync()
                    label.set(text)
                    ui.update()
                    rendered += 1
                if receiver.closed and receiver.error:
                    raise ValueError(receiver.error)
                if receiver.closed and transport.process.poll() not in (None, 0):
                    raise RuntimeError('OBSERVER_TRANSPORT_FAILED:' + str(transport.process.returncode))
                # EOF holds the last frame, clearly marked STREAM_ENDED; no replay or extrapolation.
                time.sleep(max(0., 1 / fps - (time.monotonic() - tick)))
            if calls:
                raise RuntimeError('INTEGRATION_ATTEMPT_IN_READONLY_VIEW')
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        if transport is not None:
            transport.close()
            (output / 'OBSERVER_STDERR.private.log').write_bytes(transport.stderr)
        if handle is not None:
            handle.close()
        if renderer is not None:
            renderer.close()
        if ui is not None:
            ui.destroy()
    result = dict(status='READONLY_NATIVE_OBSERVER_FAILED' if failure else 'READONLY_NATIVE_OBSERVER_COMPLETE', failure=failure, episode_id=config['episode_id'],
                  native_source_sha=config['source_sha'], scene_sha256=config['scene']['sha256'],
                  spec_sha256=config['remote']['execution_spec']['sha256'], frames_received=receiver.frames,
                  native_advances_seen=receiver.advances, displayed_frames=rendered,
                  first_native_step=first_step, last_native_step=last_step, display_status_counts=dict(counts),
                  transport_exit_code=None if transport is None else transport.process.returncode, headless=headless,
                  observer_stop_requested=False if transport is None else transport.local_stop_requested,
                  source_observer_CPU_s=None if receiver.message is None else receiver.message.get('observer_cpu_s'),
                  observer_pid=None if receiver.message is None else receiver.message.get('observer_pid'),
                  display_fps_limit=fps, new_physics_integrations=0, controller_starts=0, control_publishes=0,
                  forbidden_integration_attempts=len(calls),
                  WM_updates=0, Actor_updates=0, CPU_s=time.process_time() - cpu, wall_s=time.monotonic() - started,
                  timing='Source file age and local receive age; not a synchronized one-way network-latency measurement',
                  contact_scope='Recorded native contacts; rendering forward pass is not a new contact qualification')
    write_json(output / 'RESULT.json', result)
    write_json(output / 'DISPLAY_SAMPLES.private.json', snapshots)
    if failure:
        raise RuntimeError(failure)
    return result
