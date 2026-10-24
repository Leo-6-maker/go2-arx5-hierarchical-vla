import hashlib,json,subprocess,sys,unittest
from .cold_packet_transport import frame

CHECK="import json;print(json.dumps(dict(action=P['action'],sha=P['packet_sha256'],packet_sha=hashlib.sha256(gzip.decompress(base64.b64decode(P['packet_base64']))).hexdigest())))"

class ColdDataPacketTests(unittest.TestCase):
    def test_exact_bytes_and_action(self):
        raw=b'{"text":"Unicode \\u03bc and literal $()"}\n';x=frame(CHECK,raw,action='deploy')
        r=subprocess.run([sys.executable,'-c',x['program']],input=x['stdin_bytes'],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout)
        self.assertEqual(v,dict(action='deploy',sha=hashlib.sha256(raw).hexdigest(),packet_sha=hashlib.sha256(raw).hexdigest()))
        self.assertFalse(x['automatic_retry']);self.assertFalse(x['deadline_override'])
    def test_same_exact_wire_deterministic(self):
        self.assertEqual(frame(CHECK,b'abc',action='deploy'),frame(CHECK,b'abc',action='deploy'))
    def test_truncated_corrupt_and_trailing_data_rejected_before_program(self):
        x=frame("raise RuntimeError('MUST_NOT_REACH_CONTROL_PROGRAM')",b'important packet',action='deploy')
        for data in (x['stdin_bytes'][:-1],x['stdin_bytes']+b'x',bytes([x['stdin_bytes'][0]^1])+x['stdin_bytes'][1:]):
            with self.subTest(data_length=len(data)):
                r=subprocess.run([sys.executable,'-c',x['program']],input=data,capture_output=True)
                self.assertNotEqual(r.returncode,0);self.assertNotIn(b'MUST_NOT_REACH_CONTROL_PROGRAM',r.stderr)
                self.assertLess(len(r.stderr),2000);self.assertEqual(r.stdout,b'')
    def test_known_actions(self):
        for a in ('qsub','retry',None):
            with self.assertRaises(ValueError):frame(CHECK,b'abc',action=a)
    def test_exact_types_and_valid_program(self):
        with self.assertRaises(TypeError):frame(CHECK,'text',action='deploy')
        with self.assertRaises(ValueError):frame(CHECK,b'',action='deploy')
        with self.assertRaises(SyntaxError):frame('invalid =',b'abc',action='deploy')
    def test_data_is_not_inserted_into_source(self):
        raw=(b'PAYLOAD_ONLY_'+bytes(range(256)))*4096;x=frame(CHECK,raw,action='launch')
        self.assertNotIn('PAYLOAD_ONLY_',x['program']);self.assertLess(len(x['program']),len(CHECK)+1200)
        r=subprocess.run([sys.executable,'-c',x['program']],input=x['stdin_bytes'],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['action'],'launch')

if __name__=='__main__':unittest.main()
