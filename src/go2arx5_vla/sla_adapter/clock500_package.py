"""Go2 inference packages for the two immutable CLOCK500 branches."""
from pathlib import Path

from . import adapt5k_package, scale20k_checkpoint, scale20k_package
from .checkpoint_v1 import read_pin
from .clock500 import ARMS, ENDPOINT, PARENT, PLAN
from .multitask_contract_v6 import C20_STATISTICS


SCHEMA = 'SLA_CLOCK500_GO2_TARGET_PACKAGE_V1'


class Provider(adapt5k_package.Provider):
    @staticmethod
    def read_manifest(root, manifest_sha256):
        manifest = read_pin(dict(path=str(root/'MANIFEST.private.json'),sha256=manifest_sha256))
        rows = manifest['arrays']
        hashes = ('checkpoint_sha256','comparison_sha256','parent_state_sha256','D2_sha256',
                  'MIX_sha256','D1_freeze_sha256','state_statistics_sha256','tokenizer_sha256',
                  'partition_sha256','frozen_fingerprint','initial_state_sha256',
                  'evaluation_sha256','restore_sha256','panel_sha256','native_population_sha256')
        if (manifest['schema']!=SCHEMA or manifest['plan']!=PLAN or manifest['phase']!=ENDPOINT
                or manifest['parent_phase']!=PARENT or manifest['arm'] not in ARMS or manifest['branch_updates']!=500
                or any(manifest[k] is not False for k in ('physics','hardware_allowed','native_admission',
                                                         'formal_ADAPT_parent_eligible'))
                or manifest['action_statistics_sha256']!=C20_STATISTICS
                or not all(scale20k_checkpoint.is_digest(manifest[k]) for k in hashes)
                or not scale20k_checkpoint.is_digest(manifest['parent_manifest']['sha256'])
                or len(rows)!=29 or len({r['file'] for r in rows})!=29 or len({r['key'] for r in rows})!=29
                or any(not r['key'].startswith(scale20k_package.TARGET_PREFIXES) or r['dtype']!='float32'
                       or not scale20k_checkpoint.is_digest(r['sha256'])
                       or Path(r['file']).parts!=('parameters',Path(r['file']).name) for r in rows)):
            raise ValueError('CLOCK500_GO2_PACKAGE_REQUIRES_ITS_OWN_SEALED_BRANCH_IDENTITY')
        return manifest
