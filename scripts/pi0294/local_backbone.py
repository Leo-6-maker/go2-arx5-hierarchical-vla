"""Bind the pinned existing snapshot to an official-compatible local name."""
from pathlib import Path


def local_model_name(root, bindings):
    alias = Path(root) / 'model_inputs' / 'nvidia' / 'Cosmos-Reason2-2B'
    snapshot = Path(bindings['cosmos'])
    if not alias.is_symlink() or alias.resolve(strict=True) != snapshot.resolve(strict=True):
        raise PermissionError('EXACT_PINNED_LOCAL_COSMOS_ALIAS_REQUIRED')
    # The official backbone class selector recognizes this name, while
    # Transformers treats it as local and never queries Hub tokenizer tags.
    return str(alias)
