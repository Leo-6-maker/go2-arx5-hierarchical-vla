from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .rpc_metrics import SessionMetrics
from .mp1 import driver


class FileLedger:
    def __init__(self,path):self.path=Path(path);self.role='hopper';self.transactions=0

    @contextmanager
    def transaction(self):
        self.transactions+=1
        with self.path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            value=json.loads(self.path.read_text());yield value
            self.path.write_text(json.dumps(value))


class RPCMetricsTests(unittest.TestCase):
    def test_native_entry_reads_exact_bound_role_request(self):
        from .mp1 import rpc_metrics_probe as probe_module
        module=sys.modules[probe_module.run.__module__]
        from .native_dataset import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'plans').mkdir();(root/'evidence').mkdir()
            for role in ['core','hopper']:
                binding=root/'plans'/('RPC_METRICS_CPU_BINDING_'+role+'_2.json')
                binding.write_text(json.dumps(dict(run_uuid=root.name,role=role,attempt=2,source_sha='fixture',maximum_CPU_s=600,maximum_wall_s=120,new_model_queries=0,new_physics=0)))
                manifest=dict(code=dict(vla_commit='fixture'),execution=dict(rpc_metrics_binding_sha256=sha(binding)))
                with patch.object(module,'validate',return_value=manifest),patch.object(module,'charge_cpu') as cpu,patch.object(module,'probe',return_value=dict(status='UNIT_FIXTURE_NOT_NATIVE_PASS')):
                    value=probe_module.run(root,role,2)
                    self.assertEqual(value['source_sha'],'fixture');self.assertEqual(value['role'],role);cpu.assert_called_once()
                binding.write_text('{}')
                with patch.object(module,'validate',return_value=manifest):
                    with self.assertRaises(PermissionError):probe_module.run(root,role,2)

    def test_bounded_native_CPU_probe_is_explicitly_not_closed_loop(self):
        from .mp1.rpc_metrics_probe import probe
        with tempfile.TemporaryDirectory() as d:value=probe(Path(d)/'probe')
        self.assertEqual(value['replies_while_lock_held'],80)
        self.assertEqual(value['per_reply_shared_ledger_transactions'],0)
        self.assertEqual(value['terminal_shared_ledger_transactions'],1)
        self.assertFalse(value['full_32s_model_loop_qualified'])
        self.assertFalse(value['compute_to_login_NFS_qualified'])

    def test_real_external_lock_cannot_delay_memory_only_replies(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.json';path.write_text(json.dumps(dict(generation_queries=3,optimizer_updates=0,CPU_s=5)))
            code='import fcntl,sys;h=open(sys.argv[1],"a");fcntl.flock(h,fcntl.LOCK_EX);print("LOCKED",flush=True);sys.stdin.readline()'
            process=subprocess.Popen([sys.executable,'-u','-c',code,str(path.with_suffix('.lock'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertTrue(select.select([process.stdout],[],[],5)[0]);self.assertEqual(process.stdout.readline().strip(),'LOCKED')
                with path.with_suffix('.lock').open('a') as handle:
                    with self.assertRaises(BlockingIOError):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                ledger=FileLedger(path);metrics=SessionMetrics();replies=[]
                for i in range(80):metrics.reply(dict(sequence=i),replies.append)
                self.assertIsNone(process.poll());self.assertEqual(len(replies),80);self.assertEqual(ledger.transactions,0)
                self.assertEqual(json.loads(path.read_text())['CPU_s'],5)
                process.communicate('\n',timeout=5)
                result=metrics.settle(ledger,2.5)
                value=json.loads(path.read_text());self.assertEqual(value,dict(generation_queries=3,optimizer_updates=0,CPU_s=7.5,RPC_calls=80))
                self.assertEqual(ledger.transactions,1);self.assertTrue(result['telemetry_settled_once'])
                with self.assertRaises(PermissionError):metrics.settle(ledger,2.5)
                self.assertEqual(ledger.transactions,1)
            finally:
                if process.poll() is None:process.kill();process.communicate(timeout=5)

    def test_uncertain_settlement_or_emit_is_not_retried(self):
        class Broken:
            role='core'
            @contextmanager
            def transaction(self):
                yield {}
                raise ConnectionError('COMMIT_RESULT_UNKNOWN')
        metrics=SessionMetrics()
        with self.assertRaises(ConnectionError):metrics.reply({},lambda v:(_ for _ in ()).throw(ConnectionError()))
        self.assertEqual(metrics.operations,1)
        with self.assertRaises(ConnectionError):metrics.settle(Broken(),1.)
        with self.assertRaises(PermissionError):metrics.settle(Broken(),1.)
        with self.assertRaises(PermissionError):metrics.reply({},lambda v:None)
        for cpu in [-1,float('inf'),float('nan'),'1']:
            with self.assertRaises(ValueError):SessionMetrics().settle(Broken(),cpu)

    def test_poison_before_case_stops_both_roles_without_starting_physics(self):
        module=sys.modules[driver.routes.__module__];writes=[];starts=[]
        remote=lambda role,code,values,label:writes.append((role,values['SESSION'])) or {'status':'readback'}
        with tempfile.TemporaryDirectory() as d:
            ready=dict(run_uuid='fixture');registry=dict(cases={'one':{'panel_index':0}})
            with patch.object(module,'bind',return_value=registry):
                result=driver.routes(remote,Path(d),'panel',ready,lambda *a:starts.append(a),SimpleNamespace(sessions={'hopper':SimpleNamespace(failed=True)}))
        self.assertTrue(result['abort_remaining_sessions']);self.assertEqual(starts,[]);self.assertEqual(writes,[('core','panel'),('hopper','panel')])
