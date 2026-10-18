from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertIs(Endpoint.invoke,OriginalEndpoint.invoke)
        self.assertIs(Endpoint.guard,OriginalEndpoint.guard)
        self.assertIs(Pump.poll,OriginalPump.poll)
        self.assertIs(Pump.completed,OriginalPump.completed)

    def test_no_fixture_can_start_without_typed_pv1_native_entry(self):
        with patch('scripts.pi0294.pv1.transport_fixture.validate_runtime',side_effect=PermissionError('DATA_NOT_READY')):
            with self.assertRaisesRegex(PermissionError,'DATA_NOT_READY'):
                Endpoint('/not-created','core',BASE+'-ALLOCATION-01')
