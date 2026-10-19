"""CPU liveness tests: no model, scheduler, RPC transport or clock extension."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import control_heartbeat as heartbeat
from scripts.pi0294.mp1 import owner_heartbeat as original
from scripts.pi0294.test_mp1_transport_fixture import put


class ControlHeartbeat(unittest.TestCase):
    def test_old_capsules_do_not_silently_change_semantics(self):
        self.assertFalse(heartbeat.owner_only('/not-opened',dict(execution={})))

    def test_explicit_both_stage_and_manifest_opt_in_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=dict(owner='EXISTING_LOCAL_PC',authority=dict(protocol_ref=heartbeat.AUTHORITY),
                execution=dict(owner_heartbeat_policy=heartbeat.POLICY))
            put(root/'PV1_STAGE_BINDING.json',dict(owner_heartbeat_policy=heartbeat.POLICY,authority_release=heartbeat.AUTHORITY),True)
            self.assertTrue(heartbeat.owner_only(root,manifest))
            put(root/'PV1_STAGE_BINDING.json',dict(owner_heartbeat_policy='OTHER'),True)
            with self.assertRaises(PermissionError):heartbeat.owner_only(root,manifest)

    def test_calls_cannot_renew_owner_heartbeat_or_mask_owner_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'OWNER_HEARTBEAT.json';put(path,dict(run_uuid='owner',observed_epoch=1000))
            raw=path.read_bytes();mtime=path.stat().st_mtime_ns
            def proof(root,run):
                return original.read_live_owner(root,run,now=clock[0],synchronize=lambda p:None)
            clock=[1001]
            with patch.object(heartbeat,'read_live_owner',side_effect=proof):
                for value in (1001,1060,1299,1300):
                    clock[0]=value;self.assertFalse(heartbeat.verify_without_renewing(root,'owner')['heartbeat_written'])
                clock[0]=1300.001
                with self.assertRaises(original.HeartbeatLost):heartbeat.verify_without_renewing(root,'owner')
            self.assertEqual(path.read_bytes(),raw);self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_forged_or_extended_guard_is_not_accepted(self):
        for proof in (dict(heartbeat_written=True,watchdog_s=300),dict(heartbeat_written=False,watchdog_s=301)):
            with patch.object(heartbeat,'read_live_owner',return_value=proof):
                with self.assertRaises(PermissionError):heartbeat.verify_without_renewing('root','owner')

    def test_actual_rpc_hook_preserves_legacy_writer_and_no_new_mode_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(heartbeat,'write') as writer,patch.object(heartbeat,'verify_without_renewing',return_value='ORIGINAL300S') as reader:
                self.assertIsNone(heartbeat.keepalive(root,'core',readonly_owner=True));writer.assert_not_called();reader.assert_not_called()
                self.assertIsNone(heartbeat.keepalive(root,'hopper',readonly_owner=False));writer.assert_called_once();reader.assert_not_called()
                writer.reset_mock()
                self.assertEqual(heartbeat.keepalive(root,'hopper',readonly_owner=True),'ORIGINAL300S')
                reader.assert_called_once_with(root,root.name);writer.assert_not_called()
            with self.assertRaises(PermissionError):heartbeat.keepalive(root,'hopper',readonly_owner=1)
