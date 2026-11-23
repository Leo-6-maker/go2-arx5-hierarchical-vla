"""Revision boundary for previously admitted wire-grid references. No I/O or actuator access."""

import numpy as np
from reference_manager_v2 import ReferenceManagerV2


class PersistentReference(ReferenceManagerV2):
    def revise(self, new, now, epoch):
        """Protect past and already installed lookahead, including one join segment.

        Callers certify each independently proposed future before revision. This
        method only checks clock, provenance and prefix equality, including width.
        """
        if (
            self.terminal
            or self.pending is not None
            or self.installed is None
            or epoch != self.epoch
            or new.identity != self.intent.identity
            or not np.isfinite(now)
            or now < max(self.last, self.revision_clock)
            or now - self.last >= 0.4 - 1e-10
        ):
            self.abort()
            raise ValueError("revision without current matching installation")
        # The site commits an entire 41-point preview per installation. Protect the
        # next segment too so interpolation on the last committed interval is stable.
        stop = self.last - self.start + 2.05
        grid = np.arange(int(np.ceil(stop / 0.05 - 1e-10)) + 1) * 0.05
        old_batch = self.intent.at(grid)
        proposed = new.at(grid)
        if not all(
            np.array_equal(x, y) for x, y in zip(old_batch, proposed, strict=True)
        ):
            raise ValueError("committed prefix overwritten")
        self.intent = new
        self.revision_clock = now
        self.revision += 1
