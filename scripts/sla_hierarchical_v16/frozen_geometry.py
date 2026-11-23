"""Independent arm kinematics and collision checks on sealed MJB geometry."""

from itertools import combinations
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation


def pose(r=None, p=None):
    T = np.eye(4)
    if r is not None:
        T[:3, :3] = r
    if p is not None:
        T[:3, 3] = p
    return T


def orientation(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1)).as_matrix()


class Scene:
    def __init__(self, file, layout, expected):
        import fcl
        from audit_sealed import pin

        pin(file, expected)
        self.sha = expected
        types = {"int": "<i4", "mjtNum": "<f8", "char": "S1", "mjtByte": "u1"}
        for line in subprocess.check_output(
            [str(layout), str(file)], text=True
        ).splitlines():
            name, kind, start, rows, cols = line.split()
            if (
                kind in types
                and (
                    name.startswith(
                        ("body_", "jnt_", "geom_", "site_", "name_", "exclude_")
                    )
                    or name in ("names", "qpos0")
                )
                and int(rows) * int(cols)
            ):
                arr = np.memmap(
                    file,
                    mode="r",
                    dtype=types[kind],
                    offset=int(start),
                    shape=(int(rows), int(cols)),
                ).copy()
                setattr(self, name, arr.ravel() if int(cols) == 1 else arr)
        self.names = self.names.tobytes()

        def names(kind):
            return [
                self.names[v:].split(b"\0", 1)[0].decode() if v >= 0 else ""
                for v in getattr(self, "name_" + kind + "adr")
            ]

        self.body, self.joint, self.geom, self.site = map(
            names, ("body", "jnt", "geom", "site")
        )
        self.mount = self.body.index("x5_base_link")
        self.arm = [self.joint.index("x5_joint" + str(i)) for i in range(1, 7)]
        self.adr = self.jnt_qposadr[self.arm]
        self.fingers = self.jnt_qposadr[
            [self.joint.index("task_left_slide"), self.joint.index("task_right_slide")]
        ]
        self.tool = self.site.index("x5_ee")
        self.rest = [
            pose(orientation(q), p) for p, q in zip(self.body_pos, self.body_quat)
        ]
        self.gfix = [
            pose(orientation(q), p) for p, q in zip(self.geom_pos, self.geom_quat)
        ]
        self.sitefix = pose(
            orientation(self.site_quat[self.tool]), self.site_pos[self.tool]
        )
        self.moving = {self.body.index("x5_link" + str(i)) for i in range(1, 9)}
        self.colliders = {}
        for i, (kind, size) in enumerate(zip(self.geom_type, self.geom_size)):
            if not (self.geom_contype[i] or self.geom_conaffinity[i]):
                continue
            if kind == 0:
                shape = fcl.Halfspace(np.asarray([0.0, 0.0, 1.0]), 0.0)
            elif kind == 2:
                shape = fcl.Sphere(size[0])
            elif kind == 3:
                shape = fcl.Capsule(size[0], 2 * size[1])
            elif kind == 5:
                shape = fcl.Cylinder(size[0], 2 * size[1])
            elif kind == 6:
                shape = fcl.Box(*(2 * size))
            else:
                raise ValueError("unsupported active geometry " + str(kind))
            self.colliders[i] = fcl.CollisionObject(shape)
        excluded = set(self.exclude_signature.tolist())
        self.pairs = []
        for i, j in combinations(self.colliders, 2):
            a, b = self.geom_bodyid[[i, j]]
            if not ({a, b} & self.moving):
                continue
            x, y = sorted((int(a), int(b)))
            wa, wb = self.body_weldid[[a, b]]
            if (
                wa == wb
                or (x << 16) + y in excluded
                or (
                    wa
                    and wb
                    and (
                        self.body_weldid[self.body_parentid[wa]] == wb
                        or self.body_weldid[self.body_parentid[wb]] == wa
                    )
                )
            ):
                continue
            if (self.geom_contype[i] & self.geom_conaffinity[j]) or (
                self.geom_contype[j] & self.geom_conaffinity[i]
            ):
                self.pairs.append((i, j))

    def forward(self, q, baseline=None):
        worlds = [None] * len(self.body) if baseline is None else list(baseline)
        worlds[0] = np.eye(4)
        axes, centers = [], []
        for i in range(1, len(worlds)):
            if baseline is not None and i not in self.moving:
                continue
            T = worlds[self.body_parentid[i]] @ self.rest[i]
            start = self.body_jntadr[i]
            for j in range(start, start + self.body_jntnum[i]):
                adr = self.jnt_qposadr[j]
                kind = self.jnt_type[j]
                if kind == 0:
                    T = pose(orientation(q[adr + 3 : adr + 7]), q[adr : adr + 3])
                    continue
                delta = q[adr] - self.qpos0[adr]
                axis = self.jnt_axis[j]
                if j in self.arm:
                    axes.append(T[:3, :3] @ axis)
                    centers.append(T[:3, :3] @ self.jnt_pos[j] + T[:3, 3])
                if kind == 2:
                    T = T @ pose(p=axis * delta)
                elif kind == 3:
                    R = Rotation.from_rotvec(delta * axis).as_matrix()
                    T = T @ pose(R, self.jnt_pos[j] - R @ self.jnt_pos[j])
                else:
                    raise ValueError("unsupported articulated joint")
            worlds[i] = T
        tcp = worlds[self.site_bodyid[self.tool]] @ self.sitefix
        axes = np.array(axes)
        jac = np.vstack([np.cross(axes, tcp[:3, 3] - np.array(centers)).T, axes.T])
        return worlds, tcp, jac

    def contact(self, worlds, update_all=False):
        import fcl

        req = fcl.CollisionRequest()
        for i, obj in self.colliders.items():
            body = self.geom_bodyid[i]
            if update_all or body in self.moving:
                m = worlds[body] @ self.gfix[i]
                obj.setTransform(fcl.Transform(m[:3, :3], m[:3, 3]))
        for i, j in self.pairs:
            if fcl.collide(
                self.colliders[i], self.colliders[j], req, fcl.CollisionResult()
            ):

                def label(k):
                    return self.geom[k] or (
                        self.body[self.geom_bodyid[k]] + "#" + str(k)
                    )

                return [label(i), label(j)]
        return None

    def witness(self, c, ts, poses, widths):
        from audit_sealed import check

        actual = c["query"]["actual_state"]
        q = np.asarray(actual["qpos"]).copy()
        home, tcp, _ = self.forward(q)
        check(np.max(abs(tcp - actual["tcp"])) < 2e-6, "actual TCP reconstruction")
        check(
            np.max(abs(home[self.mount] - actual["base_pose"])) < 2e-6,
            "actual mount reconstruction",
        )
        for color, entry in c["spec"]["multitask_geometry"]["objects"].items():
            check(
                np.allclose(
                    home[self.body.index(entry["body"])],
                    actual["objects"][color]["pose"],
                    rtol=0,
                    atol=2e-6,
                ),
                "box pose",
            )
        first_contact = self.contact(home, True)
        limit = self.jnt_range[self.arm]
        result = dict(
            checked=0,
            all_samples=len(ts),
            first_failure=None,
            minimum_joint_margin_rad=None,
            original_measured_contact_pair=first_contact,
            active_pair_count=len(self.pairs),
            scene_sha256=self.sha,
        )
        for i, (t, target, width) in enumerate(zip(ts, poses, widths)):
            q[self.fingers] = width
            for _ in range(25):
                world, ach, J = self.forward(q, home)
                diff = np.r_[
                    target[:3, 3] - ach[:3, 3],
                    Rotation.from_matrix(target[:3, :3] @ ach[:3, :3].T).as_rotvec(),
                ]
                if np.max(abs(diff)) < 2e-6:
                    break
                dq = np.linalg.solve(J.T @ J + 1e-8 * np.eye(6), J.T @ diff)
                q[self.adr] += dq / max(1.0, np.linalg.norm(dq) / 0.12)
            else:
                result["first_failure"] = dict(
                    check="IK_RESIDUAL",
                    time_s=t,
                    observed=float(np.max(abs(diff))),
                    threshold=2e-6,
                )
                break
            margin = float(
                np.min(np.minimum(q[self.adr] - limit[:, 0], limit[:, 1] - q[self.adr]))
            )
            m = result["minimum_joint_margin_rad"]
            result["minimum_joint_margin_rad"] = margin if m is None else min(m, margin)
            pair = self.contact(world)
            if margin <= 0 or pair is not None:
                result["first_failure"] = dict(
                    check="JOINT_LIMIT" if margin <= 0 else "CONTACT",
                    time_s=t,
                    margin_rad=margin,
                    threshold=0,
                    geom_pair=pair,
                )
                break
            result["checked"] = i + 1
        result["pass"] = result["checked"] == len(ts) and first_contact is None
        return result
