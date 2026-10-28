"""Loopback-only PC replay server; reads allowlisted local episode files."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit
from .records import Episode, add_offline_predictions, add_local_skeleton


def handler(episodes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log private paths or source metadata.

        def do_GET(self):
            host = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed or (self.headers.get('Origin') and self.headers['Origin'] not in {'http://'+h for h in allowed}):
                self.send_error(403, 'Local same-origin viewing only')
                return
            path = urlsplit(self.path).path
            try:
                content_type = 'application/json; charset=utf-8'
                if path == '/':
                    raw = Path(__file__).with_name('viewer.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif path == '/api/episodes':
                    raw = json.dumps([{'id': e.id, 'task': e.spec['task'], 'source_kind': e.kind,
                                       'fixture': e.fixture, 'frames': len(e.frames)} for e in episodes.values()]).encode()
                elif path.startswith('/api/episode/'):
                    raw = json.dumps(episodes[path.split('/')[-1]].payload(), allow_nan=False).encode()
                elif path.startswith('/api/asset/'):
                    _, _, _, eid, index, cam = path.split('/')
                    raw = episodes[eid].asset(int(index), cam)
                    content_type = 'image/png'
                else:
                    raise KeyError()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(raw)
            except (ValueError, KeyError, IndexError, OSError):
                self.send_error(404, 'Saved record missing or identity check failed')
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--check-only', action='store_true', help='Verify saved records and every referenced RGB; do not serve')
    p.add_argument('--allow-local-kinematics', action='store_true', help='Explicitly load configured identity-checked local model; mj_forward only')
    a = p.parse_args()
    specs = json.loads(a.config.read_text())['episodes']
    episodes = {}
    for spec in specs:
        e = Episode(spec)
        if spec.get('offline_predictions'):
            add_offline_predictions(e, spec['offline_predictions'])
        if spec.get('kinematic_model'):
            if not a.allow_local_kinematics:
                raise PermissionError('LOCAL_MODEL_LOADING_REQUIRES_EXPLICIT_FLAG')
            add_local_skeleton(e, spec['kinematic_model'])
        if e.id in episodes:
            raise ValueError('DUPLICATE_EPISODE_ID')
        episodes[e.id] = e
    if not episodes:
        raise ValueError('NO_LOCAL_EPISODES')
    if a.check_only:
        checked = 0
        for episode in episodes.values():
            for i, cam in episode.assets:
                episode.asset(i, cam)
                checked += 1
        print(json.dumps({'status': 'PASS_LOCAL_SAVED_RECORD_IDENTITY_ONLY', 'episodes': len(episodes), 'RGB_files_checked': checked,
                          'actual_native_qualification': False, 'new_integrations': 0, 'new_queries': 0,
                          'fixture_episodes': sum(e.fixture for e in episodes.values())}))
        return
    with ThreadingHTTPServer(('127.0.0.1', a.port), handler(episodes)) as http:
        print(f'PC local replay: http://127.0.0.1:{http.server_port} ({len(episodes)} saved episodes)', flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
