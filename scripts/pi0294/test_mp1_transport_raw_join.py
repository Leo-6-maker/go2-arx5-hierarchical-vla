from copy import deepcopy
import json
import unittest

from . import test_mp1_pre_model_qualification as contract_tests
from .mp1 import native_transport_qualification as native
from .mp1 import pre_model_qualification as qualification
from .native_dataset import sha


class RawTransportJoinTests(unittest.TestCase):
    def setUp(self):
        self.fixture=contract_tests.PreModelQualificationContractTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.fixture.put()
        self.root,self.directory,self.path=self.fixture.root,self.fixture.directory,self.fixture.path

    def value(self):return json.loads(self.path.read_text())

    def overwrite(self,path,value):
        path.chmod(0o600);path.write_text(json.dumps(value)+'\n');path.chmod(0o400)

    def reseal(self,value):
        binding_path=self.root/value['transport']['raw_binding_relative'];binding=json.loads(binding_path.read_text())
        for member in binding['members']:binding['members'][member]=sha(self.root/member)
        self.overwrite(binding_path,binding);value['transport']['raw_binding_sha256']=sha(binding_path);self.overwrite(self.path,value)

    def test_real_raw_packet_is_required_not_just_cycle_pass_flags(self):
        value=self.value();folder,_,_=native.namespace(self.root,'N1-PREMODEL-ALLOCATION-06')
        (folder/'outbox/000.json').unlink()
        with self.assertRaises(FileNotFoundError):qualification.require(self.root,self.directory,self.fixture.allocation)

    def test_resealed_empty_wire_or_changed_request_still_fails(self):
        value=self.value();folder,_,_=native.namespace(self.root,'N1-PREMODEL-ALLOCATION-06')
        completed=json.loads((folder/'completed/000.json').read_text());completed['cycle']['wire_calls']=[]
        self.overwrite(folder/'completed/000.json',completed);value['transport']['cycles'][0]=completed['cycle'];self.reseal(value)
        with self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.fixture.allocation)

    def test_resealed_nonexpired_writer_cannot_masquerade_as_true_owner_loss(self):
        value=self.value();folder,_,_=native.namespace(self.root,'N1-PREMODEL-ALLOCATION-06')
        expired=folder/'faults/actual_expired_heartbeat/OWNER_HEARTBEAT.json';self.overwrite(expired,dict(run_uuid=self.root.name,observed_epoch=1599))
        writer=folder/'faults/actual_expired_heartbeat/WRITER_TERMINAL.json';v=json.loads(writer.read_text());v.update(heartbeat_sha256=sha(expired),observed_epoch=1599.1);self.overwrite(writer,v)
        self.reseal(value)
        with self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.fixture.allocation)

    def test_actual_core_process_closure_and_install_receipts_cannot_be_omitted(self):
        value=self.value();folder,_,_=native.namespace(self.root,'N1-PREMODEL-ALLOCATION-06')
        owner=folder/native.OWNER;v=json.loads(owner.read_text());v['nominal_terminals']['core']['native_process_terminal_verified']=False
        self.overwrite(owner,v);self.reseal(value)
        with self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.fixture.allocation)

    def test_measured_reference_load_cannot_be_replaced_with_padding_estimate(self):
        value=self.value();value['transport']['reference_RGB_payload_bytes']+=1
        self.overwrite(self.path,value)
        with self.assertRaises(PermissionError):qualification.require(self.root,self.directory,self.fixture.allocation)


if __name__=='__main__':unittest.main()
