"""CPU-only command/ownership contracts; no SSH process or socket connection."""
import os,stat,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from scripts.pi0294.dec1.owner_mux import mux_argv,checked_master

class Tests(unittest.TestCase):
    def argv(self,site):return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=/old',site,'/usr/bin/python3 -']
    def test_preserves_both_canonical_host_identities_and_deadlines(self):
        for site in ['hopper','vanda-simon']:
            a=self.argv(site);b=mux_argv(a,'/CPU_FIXTURE/control.sock')
            self.assertEqual(b[-2:],a[-2:]);self.assertEqual(b.count('ControlMaster=no'),1)
            self.assertIn('ControlPath=/CPU_FIXTURE/control.sock',b);self.assertIn('ProxyCommand=false',b)
            self.assertIn('CanonicalizeHostname=no',b)
            for option in ['BatchMode=yes','ConnectTimeout=15','StrictHostKeyChecking=yes','ServerAliveInterval=10','ServerAliveCountMax=3']:self.assertIn(option,b)
            self.assertNotIn('ControlPath=/old',b)
    def test_unknown_host_and_socket_injection_rejected(self):
        for a,p in [(self.argv('unknown'),'/tmp/s'),(self.argv('hopper'),'relative'),(self.argv('hopper'),'/tmp/a\n-b')]:
            with self.subTest(a=a,p=p),self.assertRaises(ValueError):mux_argv(a,p)
    def test_local_master_check_and_no_reconnect(self):
        parent=SimpleNamespace(st_uid=os.getuid(),st_mode=0o40700);sock=SimpleNamespace(st_uid=os.getuid(),st_mode=stat.S_IFSOCK|0o600)
        with patch.object(Path,'stat',return_value=parent),patch.object(Path,'lstat',return_value=sock):
            runner=Mock(return_value=SimpleNamespace(returncode=0));b=checked_master(self.argv('hopper'),'/CPU_FIXTURE/control.sock',runner=runner)
            runner.assert_called_once_with(['ssh','-S','/CPU_FIXTURE/control.sock','-O','check','hopper'],capture_output=True,text=True,timeout=5)
            self.assertIn('ProxyCommand=false',b)
            runner=Mock(return_value=SimpleNamespace(returncode=255))
            with self.assertRaises(ConnectionError):checked_master(self.argv('hopper'),'/CPU_FIXTURE/control.sock',runner=runner)
            self.assertEqual(runner.call_count,1)
    def test_foreign_public_or_non_socket_rejected_before_process(self):
        for uid,pmode,smode in [(os.getuid()+1,0o40700,stat.S_IFSOCK),(os.getuid(),0o40755,stat.S_IFSOCK),(os.getuid(),0o40700,stat.S_IFREG)]:
            runner=Mock()
            with patch.object(Path,'stat',return_value=SimpleNamespace(st_uid=uid,st_mode=pmode)),patch.object(Path,'lstat',return_value=SimpleNamespace(st_uid=os.getuid(),st_mode=smode)),self.assertRaises(PermissionError):checked_master(self.argv('hopper'),'/CPU_FIXTURE/control.sock',runner=runner)
            runner.assert_not_called()

if __name__=='__main__':unittest.main()
