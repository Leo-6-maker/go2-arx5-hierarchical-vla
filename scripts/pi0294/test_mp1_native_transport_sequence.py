from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import write
from scripts.pi0292.relay_protocol import digest
from .mp1 import native_transport_qualification as qualification
from .native_dataset import sha


class PlannedDiagnosticStop(Exception):
    pass


class NativeTransportSequence(unittest.TestCase):
    def test_blocking_native_faults_wait_for_all_real_nominal_cycles_and_owner_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'source-pair-01';directory=root/'pbs/alloc-07';directory.mkdir(parents=True)
            label='N1-PREMODEL-ALLOCATION-07';folder=root/'engineering/n1-transport'/label;folder.mkdir(parents=True)
            registry=root/'plans'/label/'REGISTRY.json';registry.parent.mkdir(parents=True);write(registry,{'fixture':'CPU_ONLY'})
            job='123.hopper-m-02';source='a'*40
            endpoint=SimpleNamespace(registry_path=registry,manifest={'code':{'vla_commit':source}},
                registry={'deadline_epoch':time.time()+3600},ledger=SimpleNamespace(cpu=lambda value:charges.append(value)))
            cycles=[{'sequence':i} for i in range(80)];acks=[{'fixture_ACK':i} for i in range(80)]
            (folder/'completed').mkdir()
            for i,ack in enumerate(acks):write(folder/'completed'/f'{i:03}.json',{'ack':ack})
            faults={}
            for name,suffix,code in [('late_response','-FAULT-LATE',0),('process_exit','-FAULT-EXIT',72)]:
                faults[name]=dict(status='PASS',failed_closed=True,unknown_effects_resent=0,
                    native_process_terminal_verified=True,second_call_blocked_before_send=True,actual_fault_requests=1,
                    job_id=job,source_sha=source,label=label.replace('N1-PREMODEL','N1-PREMODEL'+suffix,1),actual_server_exit=code)
            owner=dict(schema='MP1_N1_OWNER_TRANSPORT_FAULTS_V1',job_id=job,run_uuid=root.name,label=label,
                registry_sha256=sha(registry),nominal_cycles_sha256=digest(cycles),nominal_cycles=80,faults=faults,
                nominal_terminals=dict(core=dict(native_process_terminal_verified=True,actual_install_count=80,
                    actual_install_ACKs_sha256=digest(acks)),hopper=dict(native_process_terminal_verified=True)))
            events=[];charges=[]
            class Pump:
                calls=0;closed=False
                def __init__(self):self.folder=folder
                def poll(self):
                    self.calls+=1;events.append('poll')
                    if self.calls==2:
                        write(folder/qualification.OWNER,owner);(folder/qualification.OWNER).chmod(0o400)
                def completed(self):return cycles if self.calls>=2 else None
                def close(self):self.closed=True
            pump=Pump()
            def capture(*args,**kwargs):
                events.append('capture');kwargs['service']()
                return dict(rows=[],clock_source='SYNTHETIC_CPU_TEST_NOT_NATIVE_EVIDENCE')
            def finish(*args):
                self.assertGreaterEqual(pump.calls,2)
                self.assertTrue((folder/qualification.OWNER).exists())
                events.append('blocking_faults')
                raise PlannedDiagnosticStop('STOP_AFTER_ORDER_CHECK_NOT_A_NATIVE_PASS')
            allocation=dict(job_id=job,hostname='CPU_TEST_ONLY')
            with patch.object(qualification,'Endpoint',return_value=endpoint),patch.object(qualification,'Pump',return_value=pump),\
                 patch.object(qualification,'start_exit_heartbeat'),patch.object(qualification,'capture',side_effect=capture),\
                 patch.object(qualification,'finish_faults',side_effect=finish),patch.object(qualification.time,'sleep',return_value=None),\
                 patch.object(qualification.resource,'getrusage',return_value=SimpleNamespace(ru_utime=.3,ru_stime=.6)):
                with self.assertRaises(PlannedDiagnosticStop):qualification.run(root,directory,allocation,lambda:None)
            self.assertEqual(events,['capture','poll','poll','blocking_faults'])
            self.assertTrue(pump.closed);self.assertEqual(len(charges),1)
            self.assertEqual(charges,[0])
            self.assertFalse((directory/'N1_NATIVE_ENDURANCE_AND_TRANSPORT.json').exists())


if __name__=='__main__':unittest.main()
