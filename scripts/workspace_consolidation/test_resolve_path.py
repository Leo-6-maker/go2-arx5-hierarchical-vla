import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from resolve_path import resolve


class ResolverTests(unittest.TestCase):
    def test_cross_host_does_not_stat_or_connect(self):
        registry = dict(entries=[dict(role='RAW', path='/not/local', executor_node='corelab-4080s')])
        with patch.object(Path, 'exists', side_effect=AssertionError('No local SSD probe')):
            value = resolve(registry, 'RAW', 'corelab-amd-server', True)
        self.assertEqual(value['status'], 'HOST_REQUIRED')
        self.assertFalse(value['automatically_connected'])

    def test_shared_path_and_missing_role(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = dict(entries=[dict(role='CODE', path=directory, executor_node='BOTH_CORELAB_NODES')])
            self.assertEqual(resolve(registry, 'CODE', 'corelab-amd-server', True)['status'], 'PATH_OBSERVED')
            self.assertEqual(resolve(registry, 'CODE', 'corelab-4080s', True)['status'], 'PATH_OBSERVED')
            with self.assertRaises(ValueError):
                resolve(registry, 'MISSING', 'corelab-amd-server')

    def test_no_validation_means_no_stat(self):
        registry = dict(entries=[dict(role='CODE', path='/missing', executor_node='BOTH_CORELAB_NODES')])
        with patch.object(Path, 'exists', side_effect=AssertionError('Unexpected filesystem read')):
            value = resolve(registry, 'CODE', 'corelab-amd-server')
        self.assertEqual(value['status'], 'REGISTERED_NOT_LIVE_VERIFIED')

    def test_known_alias_resolves_only_verified_hostnames(self):
        registry = dict(entries=[dict(role='MODEL', path='/candidate', executor_node='hopper')],
                        validated_hostnames={'hopper': ['hopper-l-01']})
        self.assertEqual(resolve(registry, 'MODEL', 'hopper-l-01')['status'], 'REGISTERED_NOT_LIVE_VERIFIED')
        self.assertEqual(resolve(registry, 'MODEL', 'some-other-login')['status'], 'HOST_REQUIRED')


if __name__ == '__main__':
    unittest.main()
