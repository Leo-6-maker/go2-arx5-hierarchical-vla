import hashlib
import subprocess
import sys
import unittest

from .mp1.file_rpc_shutdown import close_channel


class FileRPCShutdownTests(unittest.TestCase):
    def start(self,program):
        p=subprocess.Popen([sys.executable,'-u','-c',program],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0)
        def cleanup():
            if p.poll() is None:p.terminate()
            p.wait(timeout=5)
            for stream in [p.stdin,p.stdout]:
                if stream and not stream.closed:stream.close()
        self.addCleanup(cleanup);return p

    def test_unread_response_larger_than_pipe_drains_without_recovering_an_ack(self):
        p=self.start('import sys;sys.stdout.buffer.write(b"x"*200000);sys.stdout.flush();sys.stdin.buffer.read()')
        value=close_channel(p)
        self.assertEqual(value['returncode'],0);self.assertEqual(value['status'],'PROCESS_TERMINAL')
        self.assertEqual(value['abandoned_stdout_bytes'],200000)
        self.assertEqual(value['abandoned_stdout_sha256'],hashlib.sha256(b'x'*200000).hexdigest())
        self.assertFalse(value['discarded_output_is_rpc_ack'])
        self.assertFalse(value['effects_retried']);self.assertFalse(value['process_signalled'])

    def test_still_running_process_is_unknown_not_restarted_or_terminated(self):
        p=self.start('import time;time.sleep(5)')
        value=close_channel(p,timeout=.02)
        self.assertIsNone(value['returncode']);self.assertIsNone(p.poll())
        self.assertEqual(value['status'],'UNKNOWN_PROCESS_TERMINAL')
        self.assertFalse(value['process_signalled'])

    def test_output_bound_does_not_parse_or_accept_an_oversized_response(self):
        p=self.start('import sys;sys.stdout.buffer.write(b"x"*4096);sys.stdout.flush()')
        value=close_channel(p,max_bytes=1024)
        self.assertEqual(value['status'],'ABANDONED_OUTPUT_EXCEEDS_PACKET_BOUND')
        self.assertEqual(value['abandoned_stdout_bytes'],1025)
        self.assertFalse(value['discarded_output_is_rpc_ack'])

    def test_rejects_unbounded_or_expanded_close_before_closing_stdin(self):
        p=self.start('import sys;sys.stdin.buffer.read()')
        for kwargs in [dict(timeout=0),dict(timeout=31),dict(timeout=True),dict(max_bytes=True),dict(max_bytes=2*1024**2+1)]:
            with self.assertRaises(ValueError):close_channel(p,**kwargs)
            self.assertFalse(p.stdin.closed)


if __name__=='__main__':unittest.main()
