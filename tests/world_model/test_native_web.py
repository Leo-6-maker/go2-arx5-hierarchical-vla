import http.client
from http.server import ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
import unittest

from go2arx5_vla.world_model.native_web import WebState, handler


class WebObserver(unittest.TestCase):
    def setUp(self):
        receiver = SimpleNamespace(view=lambda: dict(status='SEALED', row=dict(step=12503, time_s=25.006),
                message=dict(command=dict(stage='WITHDRAW')), frames=2, native_advances=0,
                source_state_age_s=100., receive_age_s=.1))
        config = dict(episode_id='unit-only', source_sha='source', scene=dict(sha256='scene'),
                      remote=dict(execution_spec=dict(sha256='spec')), observer_command=['private-route-never-sent-to-browser'])
        self.state = WebState(receiver, config)
        self.state.jpeg = b'unit-test-image-not-native-render'
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, path, *, method='GET', body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_page_status_and_frame_use_original_state_and_hide_private_route(self):
        code, headers, raw = self.request('/')
        self.assertEqual(code, 200)
        self.assertIn('本体实时观测'.encode(), raw)
        code, headers, raw = self.request('/api/status')
        value = json.loads(raw)
        self.assertEqual(value['status'], 'SEALED')
        self.assertEqual(value['native_step'], 12503)
        self.assertEqual(headers['X-WM02-Session'], value['session_id'])
        self.assertNotIn('private-route', raw.decode())
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['Cross-Origin-Resource-Policy'], 'same-origin')
        self.assertEqual(self.request('/api/frame.jpg')[2], self.state.jpeg)
        self.assertEqual(self.request('/api/frame.jpg')[1]['X-WM02-Session'], value['session_id'])
        other = WebState(self.state.receiver, self.state.config)
        self.assertNotEqual(other.session_id, value['session_id'])

    def test_camera_request_only_changes_view_and_rejects_control_or_bad_numbers(self):
        before = self.state.payload()
        code, _, _ = self.request('/api/view', method='POST', body=json.dumps(dict(azimuth=90, elevation=-25, distance=2)))
        self.assertEqual(code, 200)
        self.assertEqual(self.state.payload()['camera']['azimuth'], 90)
        self.assertEqual(self.state.payload()['native_step'], before['native_step'])
        self.assertEqual(self.state.camera_revision, 1)
        for data in (dict(reset=True), dict(azimuth=90, elevation=-25, distance=float('nan')),
                     dict(azimuth=90, elevation=-25, distance=100), dict(azimuth=True, elevation=-25, distance=2)):
            with self.subTest(data=data):
                self.assertEqual(self.request('/api/view', method='POST', body=json.dumps(data))[0], 400)
        self.assertEqual(self.request('/api/control', method='POST', body='{}')[0], 405)

    def test_cross_origin_host_and_arbitrary_asset_reads_are_rejected(self):
        self.assertEqual(self.request('/api/status', headers={'Host': 'external.example'})[0], 403)
        self.assertEqual(self.request('/api/view', method='POST', body='{}', headers={'Origin': 'https://external.example'})[0], 403)
        self.assertEqual(self.request('/../scene.mjb')[0], 404)
        self.assertEqual(self.request('/api/asset/scene.mjb')[0], 404)

    def test_missing_frame_and_terminal_state_are_explicit(self):
        self.state.jpeg = None
        self.assertEqual(self.request('/api/frame.jpg')[0], 503)
        self.state.finished = True
        self.assertEqual(json.loads(self.request('/api/status')[2])['status'], 'STREAM_ENDED')
        self.state.failure = 'test error'
        self.assertEqual(json.loads(self.request('/api/status')[2])['status'], 'DATA_ERROR')


if __name__ == '__main__':
    unittest.main()
