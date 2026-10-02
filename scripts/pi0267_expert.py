"""PI0267's fixed observable arclength teacher. No model or robot physics."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp
from pi0266_expert import ObservableCorrection
from pi0265_metrics import so3


class ArclengthCorrection(ObservableCorrection):
    def __init__(self, reg, family, task, command):
        super().__init__(reg, family, task, command)
        increments = np.sqrt(np.sum(np.diff(self.p, axis=0)**2, axis=1)
                             + (.05 * so3(self.r[1:], self.r[:-1]))**2)
        keep = np.r_[True, increments > 0.]
        self.ell = np.r_[0., np.cumsum(increments)][keep]
        self.source_t = self.grid[keep]
        self.p, self.r = self.p[keep], self.r[keep]
        assert len(self.ell) > 1 and np.all(np.diff(self.ell) > 0)
        self.slerp = Slerp(self.ell, Rotation.from_matrix(self.r))

    def at_length(self, length):
        length = np.clip(np.atleast_1d(length), 0., self.ell[-1])
        p = np.column_stack([np.interp(length, self.ell, self.p[:, j]) for j in range(3)])
        return p, self.slerp(length).as_matrix()

    def project(self, p, r):
        cost = np.sum((self.p-p)**2, axis=1) + (.05*so3(r, self.r))**2
        minima = np.flatnonzero((cost <= np.r_[np.inf, cost[:-1]])
                                & (cost <= np.r_[cost[1:], np.inf]))
        i = int(np.flatnonzero(cost <= cost.min()+1e-15)[0])

        def objective(length):
            cp, cr = self.at_length(length)
            return float(np.sum((cp[0]-p)**2) + (.05*so3(r, cr[0]))**2)

        def refine(j):
            lo, hi = self.ell[max(0, j-1)], self.ell[min(len(self.ell)-1, j+1)]
            fit = minimize_scalar(objective, bounds=(lo, hi), method='bounded',
                                  options=dict(xatol=1e-12, maxiter=100))
            choices = [(objective(s), float(s)) for s in (lo, self.ell[j], fit.x, hi)]
            best = min(c for c, s in choices)
            return min(((c, s) for c, s in choices if c <= best+1e-15), key=lambda x: x[1])

        # Refine every local minimum: raw grid error cannot hide a separated equal minimum.
        candidates = [refine(int(j)) for j in minima]
        if i not in minima:
            candidates.append(refine(i))
        minimum = min(c for c, s in candidates)
        equal = [(c, s) for c, s in candidates if c <= minimum+1e-15]
        length = min(s for c, s in equal)
        h = np.arange(40)*.05
        fp, fr = self.at_length(length+.02*h)
        aliases = []
        for c, other in equal:
            if abs(other-length) <= 1e-10:
                continue
            ap, ar = self.at_length(other+.02*h)
            if np.max(np.linalg.norm(ap-fp, axis=1)) > 1e-5 or np.max(so3(ar, fr)) > 1e-4:
                aliases.append(dict(ell_m=float(other), objective=float(c)))
        return length, minimum, aliases

    def label(self, p, r):
        p, r = np.asarray(p, float), np.asarray(r, float)
        assert p.shape == (3,) and r.shape == (3, 3) and np.isfinite(p).all() and np.isfinite(r).all()
        length, minimum, aliases = self.project(p, r)
        h = np.arange(40)*.05
        cp, cr = self.at_length(length+.02*h)
        decay = np.exp(-2*h)
        out_p = cp + decay[:, None]*(p-cp[0])
        residual = Rotation.from_matrix(r@cr[0].T).as_rotvec()
        out_r = Rotation.from_rotvec(decay[:, None]*residual).as_matrix()@cr
        assert np.linalg.norm(out_p[0]-p) <= 1e-12 and so3(out_r[0], r) <= 1e-10
        return out_p, out_r, dict(label_rule='ARCLENGTH_EXP_CORRECTION_V1', ell_m=float(length),
            ell_end_m=float(self.ell[-1]), source_phase_s=float(np.interp(length, self.ell, self.source_t)),
            quartile=min(3, int(length/self.ell[-1]*4)), terminal_hold=bool(length >= self.ell[-1]-1e-10),
            projection_objective=float(minimum), phase_aliases=aliases, valid_label=not aliases,
            time_or_record_id_used=False)


def self_check(reg, families):
    import copy
    checked = copy.deepcopy(reg)
    initial = np.asarray(checked['initial_TCP_world'])
    initial[:3, :3] = Rotation.from_euler('xyz', [.31, -.47, .83]).as_matrix()
    checked['initial_TCP_world'] = initial.tolist()
    rows = []
    for family in families:
        for task in ('M1', 'M3'):
            for command in (0, 1):
                teacher = ArclengthCorrection(checked, family, task, command)
                for fraction in (0., .25, .5, .75, 1.):
                    cp, cr = teacher.at_length(teacher.ell[-1]*fraction)
                    p = cp[0] + [.002, -.003, .001]
                    r = Rotation.from_rotvec([.03, -.02, .01]).as_matrix()@cr[0]
                    a, b, meta = teacher.label(p, r)
                    a2, b2, meta2 = teacher.label(p.copy(), r.copy())
                    assert np.array_equal(a, a2) and np.array_equal(b, b2) and meta == meta2
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    assert np.max(abs(b@b.transpose(0, 2, 1)-np.eye(3))) < 1e-10
                    bar_p, bar_r = teacher.at_length(meta['ell_m']+.02*np.arange(40)*.05)
                    expected_decay = np.exp(-2*np.arange(40)*.05)
                    assert np.max(abs(a-bar_p-expected_decay[:, None]*(p-bar_p[0]))) < 1e-12
                    assert np.max(abs(so3(b, bar_r)-expected_decay*so3(r, bar_r[0]))) < 1e-10
                    # Formula fixture uses endpoint-outward displacement, ensuring nearest endpoint stays terminal.
                    if fraction == 1.:
                        direction = teacher.p[-1]-teacher.p[-2]
                        direction /= np.linalg.norm(direction)
                        tp = teacher.p[-1]+.002*direction
                        tr = teacher.r[-1]
                        a, b, terminal = teacher.label(tp, tr)
                        assert abs(terminal['ell_m']-teacher.ell[-1]) < 1e-10
                        errors = np.linalg.norm(a-teacher.p[-1], axis=1)
                        assert np.all(np.diff(errors) < 0) and abs(errors[8]/errors[0]-np.exp(-.8)) < 1e-10
                    rows.append(dict(family=family['family_id'], task=task, command=command,
                                     fraction=fraction, **meta))
    return dict(status='PASS_FIXED_FORMULA_CONTINUITY_REPEAT_NONCOMMUTING_SO3_TERMINAL', rows=rows)
