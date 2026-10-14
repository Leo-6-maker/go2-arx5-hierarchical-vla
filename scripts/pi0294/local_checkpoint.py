"""Local content verification independent of remote archive availability."""
from pathlib import Path

from scripts.pi0292.durable import read, write
from .checkpoint_delivery import inventory
from .continuation import DATA, STATS, RUN, PARENT, validate_entry, within
from .native_dataset import sha


def qualify(root, checkpoint, output, *, step):
    root, checkpoint, output = Path(root), Path(checkpoint), Path(output)
    validate_entry(root, profile='hopper-neural')
    output = within(output, root)
    if output.exists() or step not in (512,2048,4096):
        raise PermissionError('NEW_LOCAL_IDENTITY_AT_REGISTERED_FORMAL_BOUNDARY_REQUIRED')
    bindings = read(root/'NEURAL_BINDINGS.private.json')
    dataset = Path(bindings['dataset'])
    parent = read(root/'PARENT_BINDING.private.json')
    expected = Path(parent['root'])/'training/main/checkpoint-512' if step == 512 else root/'training/main'/('checkpoint-'+str(step))
    if checkpoint != expected or checkpoint.is_symlink() or Path(parent['root']).name != PARENT:
        raise PermissionError('ONLY_PARENT_M512_OR_NEW_CHAIN_LOCAL_CHECKPOINT')
    if sha(dataset/'DATASET_MANIFEST.json') != DATA or sha(dataset/'statistics.json') != STATS:
        raise PermissionError('LOCAL_CHECKPOINT_ORIGINAL_V30_AND_STATS_REQUIRED')
    bound = inventory(checkpoint,dataset,expected_step=step,kind='main')
    if bound['consumed_samples'] != step*32:
        raise PermissionError('FULL_SAMPLER_GLOBAL_CURSOR_REQUIRED')
    if step == 512:
        original_path = Path(parent['root'])/'delivery/checkpoint-main-M512-01/CHECKPOINT_MANIFEST.json'
        if sha(original_path) != '6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a':
            raise PermissionError('ORIGINAL_PARENT_INVENTORY_CHANGED')
        if bound['files'] != read(original_path)['files']:
            raise PermissionError('ORIGINAL_PRIVATE_M512_BYTES_CHANGED')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'CHECKPOINT_MANIFEST.json',bound)
    identity = sha(output/'CHECKPOINT_MANIFEST.json')
    ready = dict(status='LOCAL_CHECKPOINT_READY',run_uuid=RUN,kind='main',global_step=step,
        consumed_samples=step*32,consumer_host_role='hopper-neural',inventory_sha256=identity,
        manifest_sha256=identity,files=bound['files'],full_member_SHA_verified=True,
        source_directory=str(checkpoint),remote_archive_status='INDEPENDENT_NOT_A_LOCAL_MODEL_GATE',
        model_loaded=False,pickle_executed=False,independent_neural_resume=False,
        selected='NONE',deployment='NONE')
    write(output/'LOCAL_READY.private.json',ready)
    return ready
