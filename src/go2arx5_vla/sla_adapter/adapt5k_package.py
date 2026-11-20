"""ADAPT Go2 inference packages with separate D1 state and C20 action statistics."""
import os
from pathlib import Path
import tempfile

from scripts.pi0292.durable import write
from . import adapt5k_contract, scale20k_checkpoint, scale20k_package
from .checkpoint_v1 import read_pin, sha
from .multitask_contract_v6 import C20_STATISTICS


SCHEMA = 'SLA_ADAPT0_TARGET_RGB_PACKAGE_V1'
TRAINED_SCHEMA = 'SLA_ADAPT_TRAINED_TARGET_RGB_PACKAGE_V1'
TRAINED_PHASES = (2000, 5000)


class Provider(scale20k_package.Provider):
    @staticmethod
    def read_manifest(root, manifest_sha256):
        manifest = read_pin(dict(path=str(root/'MANIFEST.private.json'),sha256=manifest_sha256))
        rows = manifest['arrays']
        schema, phase = manifest['schema'], manifest['phase']
        legacy = schema == SCHEMA and phase == 0
        trained = schema == TRAINED_SCHEMA and phase in TRAINED_PHASES
        checkpoint_key = 'ADAPT0_sha256' if legacy else 'checkpoint_sha256'
        if (not (legacy or trained) or manifest['plan'] != adapt5k_contract.PLAN
                or manifest['physics'] is not False or manifest['hardware_allowed'] is not False
                or manifest['native_admission'] is not False or manifest['action_statistics_sha256'] != C20_STATISTICS
                or not all(scale20k_checkpoint.is_digest(manifest[k]) for k in (checkpoint_key,'MIX_sha256',
                    'D1_freeze_sha256','state_statistics_sha256','tokenizer_sha256','partition_sha256',
                    'frozen_fingerprint','initial_state_sha256','evaluation_sha256','restore_sha256','panel_sha256'))
                or trained and not scale20k_checkpoint.is_digest(manifest['parent_manifest']['sha256'])
                or len(rows) != 29 or len({r['key'] for r in rows}) != 29 or len({r['file'] for r in rows}) != 29
                or any(not r['key'].startswith(scale20k_package.TARGET_PREFIXES) or r['dtype'] != 'float32'
                       or not scale20k_checkpoint.is_digest(r['sha256'])
                       or Path(r['file']).parts != ('parameters',Path(r['file']).name) for r in rows)):
            raise ValueError('ADAPT_TARGET_PACKAGE_REQUIRES_SEALED_NONPHYSICAL_IDENTITY')
        return manifest

    @staticmethod
    def load_statistics(root, manifest):
        from .adapt5k_data import STATS_SCHEMA
        from .codec_v1 import Quantiles

        state = read_pin(dict(path=str(root/'D1_STATE_STATISTICS.json'),sha256=manifest['state_statistics_sha256']))
        actions = read_pin(dict(path=str(root/'C20_ACTION_STATISTICS.json'),sha256=C20_STATISTICS))
        if (state['schema'] != STATS_SCHEMA or state['split'] != 'TRAIN'
                or state['frozen_action_statistics_sha256'] != C20_STATISTICS):
            raise ValueError('ADAPT0_TARGET_REQUIRES_D1_STATE_AND_ORIGINAL_C20_ACTION_QUANTILES')
        q_state, q_action = state['state'],actions['actions']
        return dict(state=Quantiles(q_state['q01'],q_state['q99']),actions=Quantiles(q_action['q01'],q_action['q99']))


def export(path, spec):
    """Materialize ADAPT0 locally from its independently proven identical MIX arrays."""
    from . import adapt5k_checkpoint as adapt
    from .adapt5k_data import STATS_SCHEMA

    trained = read_pin(spec['MIX_model_at_start'])
    binding = trained['binding']
    mix_root,mix = scale20k_checkpoint.read_complete(spec['MIX_checkpoint'],'SCALE-MIX20K',binding)
    zero = read_pin(spec['zero_worker'])
    # ADAPT0 metadata was copied read-only from its successful site. MIX tensor
    # bytes can serve this zero-update package only if every state row matches.
    marker = read_pin(spec['zero_complete_marker'])
    state = read_pin(spec['D1_statistics'])
    exact = read_pin(spec['zero_restore_proof'])
    evaluation = read_pin(spec['zero_formal'])
    cp = zero['checkpoint']
    if (path.exists() or zero['status'] != 'ACTUAL_ADAPT_ZERO_GRADIENTS_SEALED' or zero['exit_code']
            or zero['new_updates'] or zero['native_requests'] or zero['unchanged_parent_state_arrays'] != 140
            or cp['sha256'] != spec['zero_complete_marker']['sha256'] or marker['schema'] != adapt.SCHEMA
            or marker['binding'] != zero['binding'] or marker['phase_step'] != 0
            or marker['progress'] != adapt.identity(zero['binding'],0) or marker['arrays'] != mix['arrays']
            or zero['binding']['parent_checkpoint_sha256'] != spec['MIX_checkpoint']['sha256']
            or zero['binding']['parent_step'] != mix['step'] or mix['step'] != 500
            or zero['binding']['target_D1_sha256'] != spec['D1_freeze_sha256']
            or zero['binding']['target_state_sha256'] != spec['D1_statistics']['sha256']
            or state['schema'] != STATS_SCHEMA or state['split'] != 'TRAIN'
            or state['frozen_action_statistics_sha256'] != C20_STATISTICS
            or spec['action_statistics']['sha256'] != C20_STATISTICS
            or spec['tokenizer']['sha256'] != zero['binding']['tokenizer_sha256']
            or spec['parent_manifest']['sha256'] != zero['binding']['parent_manifest_sha256']
            or evaluation['checkpoint'] != cp or exact['checkpoint'] != cp
            or evaluation['panel']['sha256'] != spec['panel_sha256'] or exact['panel'] != evaluation['panel']
            or evaluation['status'] != 'ADAPT_OFFLINE_RESULTS_SEALED' or evaluation['role'] != 'formal'
            or evaluation['exit_code'] or evaluation['old_TEST_access'] or evaluation['native_requests']
            or evaluation['new_updates'] or evaluation['model_generations'] != 376
            or exact['schema'] != 'SLA_ADAPT5K_PHASE_ZERO_OUTPUT_RESTORE_ACTUAL_V1'
            or exact['ordinary_predictions'] != 288 or exact['independent_restore_exact'] != 48
            or exact['independent_restore_predictions'] != 48 or exact['native_requests'] or exact['optimizer_updates']):
        raise ValueError('ADAPT0_EXPORT_REQUIRES_ACTUAL_PHASE_ZERO_AND_INDEPENDENT_RESTORE_PLUS_IDENTICAL_MIX_STATE')

    work = Path(tempfile.mkdtemp(prefix=path.name+'.pending.',dir=path.parent))
    (work/'parameters').mkdir()
    rows = []
    for row in scale20k_package.checkpoint_parameters(mix):
        name = f"parameters/{row['file']}"
        scale20k_package.checked_copy(mix_root/row['file'],work/name,row['sha256'])
        rows.append(dict(row,file=name))
    for asset,name in (('D1_statistics','D1_STATE_STATISTICS.json'),('action_statistics','C20_ACTION_STATISTICS.json'),
                       ('tokenizer','paligemma_tokenizer.model')):
        pinned = spec[asset]
        scale20k_package.checked_copy(pinned['path'],work/name,pinned['sha256'])
    manifest = dict(schema=SCHEMA,plan=adapt5k_contract.PLAN,phase=0,parent_step=500,physics=False,hardware_allowed=False,
        native_admission=False,arrays=rows,parent_manifest=spec['parent_manifest'],ADAPT0_sha256=cp['sha256'],
        MIX_sha256=spec['MIX_checkpoint']['sha256'],D1_freeze_sha256=spec['D1_freeze_sha256'],
        state_statistics_sha256=spec['D1_statistics']['sha256'],action_statistics_sha256=C20_STATISTICS,
        tokenizer_sha256=spec['tokenizer']['sha256'],partition_sha256=binding['partition_sha256'],
        frozen_fingerprint=trained['frozen_fingerprint'],initial_state_sha256=binding['initial_state_sha256'],
        evaluation_sha256=spec['zero_formal']['sha256'],restore_sha256=spec['zero_restore_proof']['sha256'],
        panel_sha256=spec['panel_sha256'])
    write(work/'MANIFEST.private.json',manifest)
    digest = sha(work/'MANIFEST.private.json')
    Provider.read_manifest(work,digest)
    if path.exists():
        raise FileExistsError(path)
    os.rename(work,path)
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path),sha256=digest)



class Pool:
    """Serve the one registered ADAPT checkpoint through the original G7 IPC."""
    def __init__(self, spec):
        from go2arx5_vla.world_model import adapt5k_native as native
        from .multitask_contract_v6 import NATIVE_METHOD

        packages, bindings = spec['packages'],spec['request_bindings_by_arm']
        if (spec['scope'] != native.SCOPE or spec['plan_commit'] != native.PLAN
                or spec['goal_plan_sha'] != native.GOAL or spec['physics'] is not False
                or spec['new_physical_requests_budget'] or set(packages) != set(bindings) or len(packages) != 1
                or spec['GPU_uuid'] != spec['adapt_GPU_uuid']):
            raise PermissionError('ADAPT_NATIVE_G7_REQUIRES_ONE_NONINTEGRATING_SEALED_WORKER')
        arm, = packages
        package = packages[arm]
        manifest, _ = native.endpoint(arm,package,spec['endpoints'][arm],spec['ordinary_RGB_restores'][arm],
                                      gpu_uuid=spec['GPU_uuid'])
        binding = bindings[arm]
        if (binding['arm'] != arm or binding['policy_package_sha256'] != package['sha256']
                or binding['statistics_sha256'] != manifest['action_statistics_sha256']
                or binding['codec'] != NATIVE_METHOD or binding['compiled_population_sha256'] != native.POPULATION_SHA
                or spec['parent_manifest']['sha256'] != manifest['parent_manifest']['sha256']):
            raise PermissionError('ADAPT_NATIVE_G7_WORKER_IDENTITY_CHANGED')
        self.providers = {arm:Provider(package['path'],manifest_sha256=package['sha256'],
                                       parent_params=spec['parent_params'],
                                       parent_manifest_path=spec['parent_manifest']['path'])}
        self.selected = None

    select = scale20k_package.Pool.select
    predict = scale20k_package.Pool.predict
