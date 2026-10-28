from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import json
from scripts.pi0292.relay_protocol import sealed

from scripts.pi0294.pv1.transport_fixture import namespace,BASE,Endpoint,Pump
from scripts.pi0294.mp1.transport_fixture import Endpoint as OriginalEndpoint,Pump as OriginalPump


class TransportContracts(unittest.TestCase):
    def test_pv1_has_24_finite_namespaces_not_old_twelve_slot_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1,12,24):
                folder,number,mode=namespace(tmp,BASE+'-ALLOCATION-'+str(n).zfill(2))
                self.assertEqual(number,n);self.assertEqual(mode,BASE)
                self.assertTrue(folder.is_relative_to(Path(tmp)))
            for label in (BASE+'-ALLOCATION-25','N1-PREMODEL-ALLOCATION-01','../../outside'):
                with self.assertRaises(PermissionError):namespace(tmp,label)

    def test_packet_actions_and_freshness_primitives_are_the_mature_ones(self):
        self.assertIs(Endpoint.guard,OriginalEndpoint.guard)
        self.assertIs(Pump.completed,OriginalPump.completed)

    def test_fixture_response_uses_production_direct_open_not_negative_stat_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoint=Endpoint.__new__(Endpoint);endpoint.folder=Path(tmp);endpoint.role='hopper'
            endpoint.guard=Mock();(endpoint.folder/'outbox').mkdir()
            p=endpoint.folder/'outbox/000.json';p.write_text(json.dumps(dict(actual='recorded-response')))
            with patch.object(Path,'exists',return_value=False):
                self.assertEqual(endpoint.invoke(dict(operation='response',sequence=0)),dict(actual='recorded-response'))
            endpoint.guard.assert_called_once_with(dict(operation='response',sequence=0))

    def test_missing_response_does_not_create_or_compute_or_retry_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoint=Endpoint.__new__(Endpoint);endpoint.folder=Path(tmp);endpoint.role='hopper';endpoint.guard=Mock()
            (endpoint.folder/'outbox').mkdir()
            self.assertIsNone(endpoint.invoke(dict(operation='response',sequence=1)))
            self.assertEqual(list((endpoint.folder/'outbox').iterdir()),[])
            with patch.object(OriginalEndpoint,'invoke',return_value='mature-transaction') as invoke:
                self.assertEqual(endpoint.invoke(dict(operation='query',sequence=1)),'mature-transaction')
                invoke.assert_called_once()

    def test_compute_read_side_bypasses_cached_stat_but_keeps_unknown_effect_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Pump.__new__(Pump);p.folder=Path(tmp);p.next=0;p.guard=Mock();p.registry={};p.plan={}
            for name in ('inbox','compute-intents','outbox'):(p.folder/name).mkdir()
            query=sealed(dict(actual='same-query'))
            (p.folder/'inbox/000.json').write_text(json.dumps(query))
            with patch('scripts.pi0294.pv1.transport_fixture.receive_query',return_value=dict(sequence=0)) as receive:
                with patch('scripts.pi0294.pv1.transport_fixture.response_for',return_value=dict(actual='same-response')):
                    p.poll()
                self.assertEqual(receive.call_args.args[0],query)
            self.assertEqual(p.next,1);p.guard.assert_called_once()
            self.assertEqual(json.loads((p.folder/'outbox/000.json').read_text()),dict(actual='same-response'))
            p.next=0
            with patch('scripts.pi0294.pv1.transport_fixture.receive_query',return_value=dict(sequence=0)):
                with self.assertRaisesRegex(PermissionError,'UNKNOWN_FIXTURE_COMPUTE_EFFECT_NO_REPLAY'):p.poll()
            self.assertEqual(p.next,0)

    def test_direct_fixture_read_rejects_symlink_instead_of_following(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoint=Endpoint.__new__(Endpoint);endpoint.folder=Path(tmp);endpoint.role='hopper';endpoint.guard=Mock()
            (endpoint.folder/'outbox').mkdir();target=endpoint.folder/'target.json';target.write_text('{}')
            (endpoint.folder/'outbox/000.json').symlink_to(target)
            with self.assertRaises(PermissionError):endpoint.invoke(dict(operation='response',sequence=0))

    def test_no_fixture_can_start_without_typed_pv1_native_entry(self):
        with patch('scripts.pi0294.pv1.transport_fixture.validate_runtime',side_effect=PermissionError('DATA_NOT_READY')):
            with self.assertRaisesRegex(PermissionError,'DATA_NOT_READY'):
                Endpoint('/not-created','core',BASE+'-ALLOCATION-01')
