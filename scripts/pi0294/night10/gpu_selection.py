"""Choose a device at a cold job boundary; never move a live CUDA context.

The resident owner supplies fresh capacity and workload-specific qualification.
This function neither acquires ownership nor starts/retries any work.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    index: int
    uuid: str
    free_MiB: int
    entry_MiB: int
    qualified: bool
    shared_authorized: bool
    other_compute_present: bool
    stable_seconds: float


def choose_device(devices, *, qualification=False, available_RAM_bytes,
                  required_RAM_bytes, stability_seconds=120):
    """Prefer qualified capacity, then the largest actual spare margin.

    Native qualification may select a not-yet-qualified candidate. Formal work
    cannot. Physical owner and journal checks remain the resident's job.
    """
    if available_RAM_bytes < required_RAM_bytes:
        return None
    rows = list(devices)
    if len({d.uuid for d in rows}) != len(rows) or len({d.index for d in rows}) != len(rows):
        raise ValueError('DUPLICATE_DEVICE_IDENTITY')
    ready = [d for d in rows if d.free_MiB >= d.entry_MiB
             and (d.qualified or qualification)
             and (not d.other_compute_present or d.shared_authorized)
             and d.stable_seconds >= stability_seconds]
    return max(ready, key=lambda d: (d.qualified, d.free_MiB-d.entry_MiB, -d.index), default=None)
