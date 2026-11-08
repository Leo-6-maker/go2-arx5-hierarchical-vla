"""Tiny opaque fixtures only: no network, tensor loading or large-file reads."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.wm02 import stage_fusion_corr_asset as staging
from scripts.pi0294.dec1 import inference_assets as original


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def tiny_original_consumer(root):
    """Exercise actual consume/validate_manifest with only tiny fake asset bytes."""
    first = b'SYNTHETIC_ONLY_FIRST_SHARD_NOT_A_MODEL'
    statistics = b'{"fixture":"SYNTHETIC_ONLY_STATISTICS"}'
    nine = ['config.json', 'statistics.json', 'model.safetensors.index.json',
            'processor/embodiment_id.json', 'processor/processor_config.json',
            'processor/statistics.json', original.FIRST,
            'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors']
    payload = {name: ('SYNTHETIC_ONLY_' + name).encode() for name in nine if name != original.FIRST}
    payload['statistics.json'] = statistics
    for name in original.LICENSE_NAMES:
        payload[name] = ('SYNTHETIC_ONLY_LICENSE_' + name).encode()
    original_hashes = {name: sha(first if name == original.FIRST else payload[name]) for name in nine}
    # The original manifest requires explicit original/public metadata changes.
    original_hashes['config.json'] = sha(b'SYNTHETIC_OLD_CONFIG')
    original_hashes['processor/processor_config.json'] = sha(b'SYNTHETIC_OLD_PROCESSOR')
    dependency = dict(original.dependency(), bytes=len(first), sha256=sha(first))
    manifest = dict(schema=original.SCHEMA, authority_release=original.AUTHORITY,
        original_checkpoint_manifest_sha256=original.PARENT, statistics_sha256=sha(statistics),
        global_step=2048, dependency=dependency, full_resume=False, model_loaded=False,
        selected='NONE', deployment='NONE', optimizer_or_rng_included=False,
        files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in payload.items()},
        original_inference_sha256=original_hashes,
        metadata_changes={name: dict(original_sha256=original_hashes[name],
            change='ONLY_PRIVATE_COSMOS_MODEL_NAME_TO_PUBLIC_REPO_ID')
            for name in ('config.json', 'processor/processor_config.json')})
    manifest_raw = (json.dumps(manifest, indent=2)+'\n').encode()
    manifest_path = root/'SYNTHETIC_PINNED_MANIFEST.json'
    manifest_path.write_bytes(manifest_raw)
    source = root/'SYNTHETIC_OLD_E1_FIRST'
    source.write_bytes(first)
    source.chmod(0o440)
    job = root/'SYNTHETIC_NEW_FUSION_JOB'
    job.mkdir()
    (job/'cache').mkdir()
    output = job/'fit-01'
    output.mkdir()
    spec = dict(profile=staging.PROFILE, execution_site=staging.SITE, kind=staging.KIND,
        device='cpu', physics=False, GR00T_updates=0, WBC_changes=False,
        repo_id=staging.REPO_ID, data_revision=staging.DATA_REVISION, prefix=staging.PREFIX,
        manifest=str(manifest_path), manifest_sha256=sha(manifest_raw),
        source_first_shard=str(source), source_first_shard_sha256=sha(first), source_first_shard_bytes=len(first),
        source_sha='SYNTHETIC_ONLY_SOURCE', uid=os.getuid(), registered_root=str(job), campaign_root=str(root/'campaign'),
        fusion_binding=str(root/'fusion/control/BINDING.json'),
        pilot1_binding=str(root/'pilot/control/BINDING.json'),
        output_storage_proof='/SYNTHETIC_ONLY/QUALIFIED_PROOF', output_storage_proof_sha256='a'*64,
        deadline_epoch=time.time()+60.)
    payload['INFERENCE_MANIFEST.json'] = manifest_raw
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        name = filename[len(staging.PREFIX)+1:]
        path = Path(kwargs['local_dir'])/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload[name])
        return str(path)

    with ExitStack() as stack:
        for target, value in [(staging, dict(FIRST_BYTES=len(first), FIRST_SHA256=sha(first),
                                             MANIFEST_SHA256=sha(manifest_raw))),
                              (original, dict(FIRST_SHA=sha(first), R0=sha(statistics)))]:
            for name, item in value.items():
                stack.enter_context(patch.object(target, name, item))
        stack.enter_context(patch.object(original, 'dependency', return_value=dependency))
        stack.enter_context(patch('go2arx5_vla.world_model.pilot1_storage.qualified_block_path',
                                 side_effect=lambda path, *_: Path(path)))
        stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        yield spec, output, payload, download, calls


class CorrAssetStaging(unittest.TestCase):
    def test_uid_mapped_qualified_hdd_runs_original_consumer_without_chmod(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, _ = fixture
            spec['uid'] = 2009
            root = Path(spec['registered_root'])
            actual_stat = Path.stat
            def mapped(path, *args, **kwargs):
                value = actual_stat(path, *args, **kwargs)
                if Path(path).is_relative_to(root):
                    attributes = {name: getattr(value, name) for name in dir(value) if name.startswith('st_')}
                    attributes.update(st_uid=0, st_dev=2050, st_mode=stat.S_IFMT(value.st_mode) | 0o777)
                    return SimpleNamespace(**attributes)
                return value
            before = staging.identity(spec['source_first_shard'])
            with patch.object(os, 'getuid', return_value=2009), patch.object(Path, 'stat', mapped), \
                    patch.object(os, 'chmod', side_effect=AssertionError('NO_CHMOD')), \
                    patch.object(os, 'fchmod', side_effect=AssertionError('NO_FCHMOD')):
                report = staging.run(spec, output, download=download)
            self.assertEqual(len(report['inference_files_sha256']), 9)
            self.assertEqual(report['observed_asset_root_identity']['uid'], 0)
            self.assertEqual(report['observed_asset_root_identity']['device'], 2050)
            self.assertEqual((report['process_uid'], report['spec_uid']), (2009, 2009))
            self.assertFalse(report['staged_copy_posix_readonly_enforced'])
            self.assertEqual(staging.identity(spec['source_first_shard']), before)

    def test_actual_original_consumer_small_stream_root_local_downloads_and_old_inode_untouched(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, payload, download, calls = fixture
            before = staging.identity(spec['source_first_shard'])
            with patch.object(original, 'guard', side_effect=AssertionError('OLD_GUARD_FORBIDDEN')), \
                    patch.object(original, 'run', side_effect=AssertionError('OLD_SCOPE_RUN_FORBIDDEN')), \
                    patch.object(original, 'consume', wraps=original.consume) as consumed:
                staging.run(spec, output, download=download)
                self.assertEqual(consumed.call_count, 1)
            root = Path(spec['registered_root'])
            report = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(len(calls), 14)
            self.assertEqual(set(name.split(staging.PREFIX+'/')[1] for _, name, _ in calls), set(payload))
            for repo_id, filename, kwargs in calls:
                self.assertEqual(repo_id, staging.REPO_ID)
                self.assertEqual(kwargs, dict(repo_type='dataset', revision=staging.DATA_REVISION,
                    local_dir=root/'download', cache_dir=root/'cache', token=False))
            self.assertEqual(report['accounting']['copied_first_shard_bytes'], staging.FIRST_BYTES)
            self.assertEqual(report['accounting']['requested_file_bytes'], sum(map(len, payload.values())))
            self.assertEqual(report['accounting']['returned_download_file_bytes'], sum(map(len, payload.values())))
            self.assertEqual(report['accounting']['download_file_calls_completed'], 14)
            self.assertEqual(staging.identity(spec['source_first_shard']), before)
            local = root/'frozen-base'/staging.FIRST
            final = root/'inference'/staging.FIRST
            self.assertNotEqual(local.stat().st_ino, Path(spec['source_first_shard']).stat().st_ino)
            self.assertEqual(local.stat().st_ino, final.stat().st_ino)
            self.assertEqual(len(report['inference_files_sha256']), 9)
            self.assertTrue(report['original_first_shard_inode_mode_and_bytes_retained'])
            self.assertFalse(report['full_resume'])
            self.assertIsNone(report['network_wire_bytes'])
            self.assertEqual(report['model_loads'], 0)
            self.assertEqual(report['VLA_inferences'], 0)
            self.assertFalse(report['task_transfer_gain'])
            self.assertEqual(report['metadata_changes'].keys(), {'config.json', 'processor/processor_config.json'})
            with self.assertRaises(FileExistsError):
                staging.run(spec, output, download=download)

    def test_copy_stop_or_corrupt_source_preserves_partial_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root/'old', root/'partial'
            source.write_bytes(b'abcdefghij')
            before = staging.identity(source)
            checks = [0]
            accounting = dict(copied_first_shard_bytes=0)
            def stop():
                checks[0] += 1
                if checks[0] == 3:
                    raise TimeoutError('SYNTHETIC_STOP')
            with self.assertRaises(TimeoutError):
                staging.copy_frozen(source, destination, expected_bytes=10, expected_sha256=sha(source.read_bytes()),
                                    check=stop, accounting=accounting, chunk_bytes=3)
            self.assertEqual(destination.read_bytes(), b'abc')
            self.assertEqual(accounting['copied_first_shard_bytes'], 3)
            self.assertEqual(staging.identity(source), before)
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            Path(spec['source_first_shard']).chmod(0o600)
            Path(spec['source_first_shard']).write_bytes(b'x'*staging.FIRST_BYTES)
            with self.assertRaisesRegex(ValueError, 'FIRST_COPY_BYTES_HASH'):
                staging.run(spec, output, download=download)
            self.assertEqual(calls, [])
            report = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(report['status'], 'FAILED_OR_PARTIAL_PRESERVED_NO_AUTOMATIC_RETRY')
            self.assertTrue((Path(spec['registered_root'])/'frozen-base'/staging.FIRST).exists())

    def test_negative_manifest_pins_and_unqualified_root_have_zero_asset_effects(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            for change in [dict(manifest_sha256='bad'), dict(source_first_shard_sha256='bad'),
                           dict(data_revision='main'), dict(repo_id='other/repo'), dict(physics=True)]:
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    staging.run(dict(spec, **change), output, download=download)
            with self.assertRaisesRegex(PermissionError, 'ADMITTED_OUTPUT'):
                staging.run(dict(spec, uid=spec['uid']+1), output, download=download)
            with patch('go2arx5_vla.world_model.pilot1_storage.qualified_block_path',
                       side_effect=PermissionError('UNQUALIFIED_ROOT')):
                with self.assertRaisesRegex(PermissionError, 'UNQUALIFIED_ROOT'):
                    staging.run(spec, output, download=download)
            Path(spec['manifest']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'MANIFEST_CHANGED'):
                staging.run(spec, output, download=download)
            self.assertEqual(calls, [])
            self.assertFalse((Path(spec['registered_root'])/'ASSET_STAGE_INTENT.private.json').exists())

    def test_download_failure_has_one_attempt_preserves_bytes_and_no_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            def fail_second(*args, **kwargs):
                if calls:
                    path = Path(kwargs['local_dir'])/args[1]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'SYNTHETIC_PARTIAL')
                    raise OSError('SYNTHETIC_NETWORK_FAILURE_NO_RETRY')
                return download(*args, **kwargs)
            with self.assertRaisesRegex(OSError, 'NO_RETRY'):
                staging.run(spec, output, download=fail_second)
            root = Path(spec['registered_root'])
            report = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(report['accounting']['download_file_calls_attempted'], 2)
            self.assertEqual(report['accounting']['download_file_calls_completed'], 1)
            self.assertFalse((root/'CONSUMPTION.json').exists())
            self.assertTrue((root/'download'/staging.PREFIX/'config.json').exists())
            self.assertFalse(report['automatic_retry'])
            self.assertGreater(report['retained_download_tree_logical_bytes'],
                               report['accounting']['returned_download_file_bytes'])

    def test_cache_symlinks_stop_or_unlisted_downloads_are_rejected_before_external_effects(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            root = Path(spec['registered_root'])
            # Replacing this empty fixture directory is a test-only operation.
            (root/'cache').rmdir()
            (root/'cache').symlink_to(Path(tmp))
            with self.assertRaisesRegex(PermissionError, 'OWNED_HF_CACHE'):
                staging.run(spec, output, download=download)
            self.assertEqual(calls, [])
            self.assertFalse((root/'frozen-base').exists())
            self.assertFalse((root/'ASSET_STAGE_INTENT.private.json').exists())
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            root = Path(spec['registered_root'])
            (root/'STOP.json').write_text('{}')
            with self.assertRaisesRegex(TimeoutError, 'DEADLINE_OR_STOP'):
                staging.run(spec, output, download=download)
            self.assertEqual(calls, [])
            self.assertFalse((root/'ASSET_STAGE_INTENT.private.json').exists())
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            root = Path(spec['registered_root'])
            def rogue_consume(_root, _bindings, *, download, frozen_member_storage):
                return download(staging.REPO_ID, staging.PREFIX+'/UNLISTED_LARGE_MODEL',
                                repo_type='dataset', revision=staging.DATA_REVISION,
                                local_dir=root/'download')
            with patch.object(original, 'consume', side_effect=rogue_consume):
                with self.assertRaisesRegex(PermissionError, 'FOURTEEN_FIXED'):
                    staging.run(spec, output, download=download)
            self.assertEqual(calls, [])
            report = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(report['accounting']['download_file_calls_attempted'], 0)

    def test_read_only_asset_import_never_imports_model_libraries_or_hub(self):
        command = [sys.executable, '-B', '-c',
            'import sys; import scripts.wm02.stage_fusion_corr_asset; '
            'import scripts.pi0294.dec1.inference_assets; '
            'assert not any(n in sys.modules for n in ("torch","transformers","gr00t","huggingface_hub"))']
        subprocess.run(command, check=True, env=dict(os.environ, PYTHONPATH='src'))

    def test_original_consumer_receipt_cannot_override_later_failed_stage_readback(self):
        with tempfile.TemporaryDirectory() as tmp, tiny_original_consumer(Path(tmp)) as fixture:
            spec, output, _, download, calls = fixture
            def changed_after_last_download(*args, **kwargs):
                path = download(*args, **kwargs)
                if len(calls) == 14:
                    # Simulate a third party changing the old input; production
                    # staging never writes or changes its mode.
                    source = Path(spec['source_first_shard'])
                    source.chmod(0o600)
                    source.write_bytes(b'x'*staging.FIRST_BYTES)
                return path
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_FIRST_SHARD_CHANGED'):
                staging.run(spec, output, download=changed_after_last_download)
            self.assertTrue((Path(spec['registered_root'])/'CONSUMPTION.json').exists())
            report = json.loads((output/'RESULT.private.json').read_text())
            self.assertEqual(report['status'], 'FAILED_OR_PARTIAL_PRESERVED_NO_AUTOMATIC_RETRY')
            self.assertIn('NOT_CONSUMPTION_ALONE', report['eligibility'])


if __name__ == '__main__':
    unittest.main()
