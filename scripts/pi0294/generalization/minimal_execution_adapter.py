"""Exact old/replacement selection; no alias, dispatch, budget or permission.

Old validators keep reading only the immutable original row. New consumers
must name their request and use this join to validate its complete GLOBAL.
"""
from . import minimal_execution as original, minimal_retry_execution as retry

PHYSICAL = 'plans/MINIMAL_RETRY_PHYSICAL_CUSTODY.private.json'
PREDECESSOR = 'plans/MINIMAL_RETRY_PREDECESSOR_CLOSED.private.json'


def select(version, profile):
    adapters = {a.VERSION: a for a in (original, retry)}
    profiles=(retry.PROFILE,'vanda-arx5-minimal-raw-parent-first512-zero-effect-followup')if version==retry.VERSION else(original.PROFILE,)
    original.need(version in adapters and profile in profiles,
                  'ADAPTER_EXACT_VERSION_AND_PROFILE_NOT_OLD_ALIAS')
    return adapters[version]


def request(value, *, central=False):
    adapter = select(value.get('platform_training_version'), value.get('profile'))
    adapter.request(value, central=central)
    return adapter


def manifest(value):
    return select(value.get('platform_training_version'), value.get('execution', {}).get('profile'))


def registered(g, request_id, *, version=None):
    matches = [r for r in g['PBS_submissions'] if r.get('request_id') == request_id]
    original.need(len(matches) == 1, 'ADAPTER_EXACT_REQUEST_NOT_FIRST_ROW')
    row = matches[0]; adapter = request(row, central=True); marker = adapter.validate(g)
    original.need(marker is not None and row in adapter.rows(g)
        and (version is None or version == adapter.VERSION), 'ADAPTER_COMPLETE_ADOPTED_CONTEXT')
    return adapter, row, marker


def current(g):
    """Explicit adopted main index, including its real terminal state."""
    if retry.FIELD in g:
        return registered(g, retry.current_id(g), version=retry.VERSION)
    marker = original.validate(g)
    original.need(marker is not None and len(marker['registered_requests']) == 1,
                  'ADAPTER_ONE_ORIGINAL_MAIN_INDEX')
    return registered(g, marker['registered_requests'][0], version=original.VERSION)
