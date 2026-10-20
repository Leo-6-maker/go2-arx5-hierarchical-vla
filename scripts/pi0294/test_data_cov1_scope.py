"""Native-path admission fixtures, without engine imports or remote access."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.pi0294.data_cov1 import scope
from scripts.pi0294.data_cov1.plan import PLAN_RELEASE,CARD_SHA256,protocol_card


def adoption():
    return dict(status='DATA_COV1_NATIVE_ADOPTED',protocol_release=PLAN_RELEASE,
        protocol_card_sha256=CARD_SHA256,approval_level='FULL_CONDITIONAL_CAMPAIGN',
        owner='EXISTING_LOCAL_PC',approval_ref='EXPLICIT_USER_FIXTURE_NOT_REAL_GO',
        D0_lease_reused=False,PV1_budget_reassigned=False,
        approved_limits=deepcopy(protocol_card()['stages'][-1]['proposed_cumulative_limits']),T0_epoch=time.time()-10)


class DataCov1Scope(unittest.TestCase):
    def test_no_GO_old_lease_budget_or_clock_reassignment_is_not_authority(self):
        self.assertEqual(scope.approval(adoption())['approval_level'],'FULL_CONDITIONAL_CAMPAIGN')
        for field,value in [('approval_ref',None),('approval_level','PLAN_ONLY'),
                ('D0_lease_reused',True),('PV1_budget_reassigned',True),
                ('T0_epoch',time.time()+100),('protocol_card_sha256','changed')]:
            bad=adoption();bad[field]=value
            with self.assertRaises(PermissionError):scope.approval(bad)
        bad=adoption();bad['approved_limits']['CoreLab_CPU_core_hours']+=1
        with self.assertRaises(PermissionError):scope.approval(bad)

    def test_final_mode_hash_owner_and_symlink_are_real_not_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_text('{}');p.chmod(0o644)
            with self.assertRaises(PermissionError):scope.sealed_file(p)
            p.chmod(0o400);digest=scope.sealed_file(p)
            self.assertEqual(scope.sealed_file(p,digest),digest)
            with self.assertRaises(ValueError):scope.sealed_file(p,'changed')
            q=Path(d)/'link';q.symlink_to(p)
            with self.assertRaises(PermissionError):scope.sealed_file(q)
            with patch.object(os,'getuid',return_value=os.getuid()+1):
                with self.assertRaises(PermissionError):scope.sealed_file(p)

    def test_no_new_episode_dispatch_on_running_stale_or_stopped_PV1(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'campaign';root=parent/'children/worker';root.mkdir(parents=True)
            pv1=Path(d)/'protected-pv1';pv1.mkdir()
            value=dict(owner='EXISTING_LOCAL_PC',job_id='fixed-fixture',observed_epoch=time.time(),
                native_job_state='Q',allow_future_data_dispatch=True,PV1_native_root=str(pv1))
            path=parent/'PV1_PRIORITY_STATE.private.json'
            def put(v):path.write_text(json.dumps(v))
            put(value);self.assertEqual(scope.dispatch_priority(root)['native_job_state'],'Q')
            for field,new in [('native_job_state','R'),('observed_epoch',time.time()-181),
                    ('allow_future_data_dispatch',False),('owner','SECOND_OWNER')]:
                put(dict(value,**{field:new}))
                with self.assertRaises(PermissionError):scope.dispatch_priority(root)
            put(value);(pv1/'STOP.json').write_text('{}')
            with self.assertRaises(PermissionError):scope.dispatch_priority(root)

    def test_wrong_host_and_legacy_scope_rejected_before_any_worker_effect(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'children/worker';root.mkdir(parents=True)
            binding=dict(run_root=str(root),registered_parent=str(root.parent),uid=os.getuid(),host=os.uname().nodename)
            manifest=dict(schema=scope.SCOPE,scope=scope.SCOPE,role='STATIC',owner='EXISTING_LOCAL_PC',run_uuid=root.name,
                execution=dict(host=os.uname().nodename,uid=os.getuid(),profile='4080-simulation'))
            def put(name,value):
                p=root/name
                if p.exists():p.chmod(0o600)
                p.write_text(json.dumps(value));p.chmod(0o400)
            put('RUN_MANIFEST.json',manifest);put('INPUT_BINDINGS.private.json',{})
            put('OUTPUT_BINDING.private.json',dict(binding,host='wrong-host'))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                with self.assertRaisesRegex(PermissionError,'HOST_AND_CPU_ONLY'):scope.validate(root,'STATIC')
                put('OUTPUT_BINDING.private.json',binding)
                put('RUN_MANIFEST.json',dict(manifest,scope='PI0294_PV1_CORE_TRAIN_CAPTURE_V1'))
                with self.assertRaisesRegex(PermissionError,'HOST_AND_CPU_ONLY'):scope.validate(root,'STATIC')


if __name__=='__main__':unittest.main()
