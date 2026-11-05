"""Loopback web view of the existing read-only native state channel."""
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import uuid

from .native_live import LiveReceiver, ObserverTransport, load_scene
from .native_replay import apply_frame, forbid_integration
from .storage import write_json


class WebState:
    def __init__(self, receiver, config):
        self.receiver, self.config = receiver, config
        self.session_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.jpeg = None
        self.frame_id = 0
        self.frame_step = None
        self.metrics = {}
        self.camera = dict(azimuth=135., elevation=-23., distance=1.8)
        self.camera_revision = 0
        self.finished = False
        self.failure = None

    def set_camera(self, value):
        if set(value) != {'azimuth', 'elevation', 'distance'}:
            raise ValueError('VIEW_PARAMETERS_ONLY')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in value.values()):
            raise ValueError('FINITE_VIEW_PARAMETERS_REQUIRED')
        if not (-360 <= value['azimuth'] <= 360 and -89 <= value['elevation'] <= 30 and .5 <= value['distance'] <= 5):
            raise ValueError('VIEW_PARAMETER_RANGE')
        with self.lock:
            self.camera = dict(value)
            self.camera_revision += 1

    def payload(self):
        view = self.receiver.view()
        message, row = view['message'], view['row']
        with self.lock:
            return dict(status='DATA_ERROR' if self.failure else ('STREAM_ENDED' if self.finished else view['status']),
                        session_id=self.session_id,
                        episode_id=self.config['episode_id'], frame_id=self.frame_id, frame_step=self.frame_step,
                        native_step=None if row is None else row['step'], sim_time_s=None if row is None else row['time_s'],
                        frames_received=view['frames'], native_advances_seen=view['native_advances'],
                        source_state_age_s=view['source_state_age_s'], receive_age_s=view['receive_age_s'],
                        stage=None if message is None else message['command'].get('stage'),
                        metrics=dict(self.metrics), camera=dict(self.camera),
                        provenance=dict(native_source_sha=self.config['source_sha'], scene_sha256=self.config['scene']['sha256'],
                                        spec_sha256=self.config['remote']['execution_spec']['sha256']))


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            return (self.headers.get('Host') in hosts
                    and self.headers.get('Origin') in (None, *('http://' + h for h in hosts)))

        def reply(self, code, data, kind='application/json; charset=utf-8'):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('X-WM02-Session', state.session_id)
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.allowed():
                self.reply(403, b'{}')
                return
            path = urlsplit(self.path).path
            if path == '/':
                self.reply(200, Path(__file__).with_name('native_web.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/status':
                self.reply(200, json.dumps(state.payload(), allow_nan=False).encode())
            elif path == '/api/frame.jpg':
                with state.lock:
                    image = state.jpeg
                self.reply(200 if image else 503, image or b'', 'image/jpeg')
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            if not self.allowed():
                self.reply(403, b'{}')
                return
            if urlsplit(self.path).path != '/api/view':
                self.reply(405, b'{}')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 1 <= length <= 512:
                    raise ValueError('SMALL_VIEW_REQUEST_REQUIRED')
                state.set_camera(json.loads(self.rfile.read(length)))
                self.reply(200, b'{"ok":true}')
            except (ValueError, TypeError):
                self.reply(400, b'{}')

    return Handler


def serve(config, output, *, port=8766, max_wall=3600., fps=12., ready=None):
    """Render only in this thread; HTTP handlers can change the viewing camera only."""
    from PIL import Image
    from scripts.pi0294.task_state import ActualState
    if not 0 <= port <= 65535 or not 1 <= max_wall <= 86400 or not 1 <= fps <= 30:
        raise ValueError('FINITE_WEB_OBSERVER_REQUIRED')
    output = Path(output)
    output.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    mujoco, model, data = load_scene(config)
    receiver = LiveReceiver(dict(config, ngeom=int(model.ngeom)))
    state = WebState(receiver, config)
    actual = ActualState(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.28, 0., .45]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 800
    counts, calls, transport, http, thread = Counter(), [], None, None, None
    failure = None
    key = None
    try:
        with forbid_integration(mujoco) as calls, mujoco.Renderer(model, height=800, width=1200) as renderer:
            http = ThreadingHTTPServer(('127.0.0.1', port), handler(state))
            http.daemon_threads = True
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{http.server_port}/'
            write_json(output / 'READY.private.json', dict(url=url, pid=os.getpid(), max_wall_s=max_wall, renderer='MuJoCo3.2.7 original full-mesh scene'))
            print(json.dumps(dict(status='WM02_WEB_READY', url=url, max_wall_s=max_wall)), flush=True)
            if ready:
                ready(url, state)
            transport = ObserverTransport(config, receiver, max_wall, fps)
            while time.monotonic() - started < max_wall:
                tick = time.monotonic()
                view = receiver.view()
                counts[view['status']] += 1
                row, message = view['row'], view['message']
                with state.lock:
                    camera_values, revision = dict(state.camera), state.camera_revision
                current_key = None if row is None else (message['packet_sha256'], revision)
                if current_key is not None and current_key != key:
                    apply_frame(mujoco, model, data, row)
                    actual_state = actual.update(row)
                    for name, value in camera_values.items():
                        setattr(camera, name, value)
                    renderer.update_scene(data, camera=camera)
                    image = Image.fromarray(renderer.render())
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=87)
                    with state.lock:
                        state.jpeg = buffer.getvalue()
                        state.frame_id += 1
                        state.frame_step = row['step']
                        state.metrics = dict(tcp_m=actual_state['tcp'][:3, 3].tolist(),
                                             left_contact=actual_state['left_object_contact'],
                                             right_contact=actual_state['right_object_contact'],
                                             installed_horizon_goal_m=message['command'].get('installed_horizon_goal_m'))
                    key = current_key
                if receiver.error:
                    raise ValueError(receiver.error)
                if receiver.closed and transport.process.poll() not in (None, 0):
                    raise RuntimeError('OBSERVER_TRANSPORT_FAILED:' + str(transport.process.returncode))
                time.sleep(max(0., 1 / fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
        state.failure = failure
    finally:
        state.finished = True
        if transport is not None:
            transport.close()
            (output / 'OBSERVER_STDERR.private.log').write_bytes(transport.stderr)
        if http is not None:
            http.shutdown()
            http.server_close()
            thread.join(timeout=3)
        if state.jpeg:
            (output / 'last.private.jpg').write_bytes(state.jpeg)
    result = dict(status='READONLY_WEB_OBSERVER_FAILED' if failure else 'READONLY_WEB_OBSERVER_COMPLETE', failure=failure,
                  episode_id=config['episode_id'], native_source_sha=config['source_sha'],
                  scene_sha256=config['scene']['sha256'], spec_sha256=config['remote']['execution_spec']['sha256'],
                  frames_received=receiver.frames, rendered_frames=state.frame_id, native_advances_seen=receiver.advances,
                  display_status_counts=dict(counts), last_native_step=state.frame_step,
                  observer_pid=None if receiver.message is None else receiver.message.get('observer_pid'),
                  source_observer_CPU_s=None if receiver.message is None else receiver.message.get('observer_cpu_s'),
                  transport_exit_code=None if transport is None else transport.process.returncode,
                  observer_stop_requested=False if transport is None else transport.local_stop_requested,
                  new_physics_integrations=0, forbidden_integration_attempts=len(calls), controller_starts=0,
                  control_publishes=0, WM_updates=0, Actor_updates=0, CPU_s=time.process_time() - cpu,
                  wall_s=time.monotonic() - started)
    write_json(output / 'RESULT.json', result)
    if failure:
        raise RuntimeError(failure)
    return result
