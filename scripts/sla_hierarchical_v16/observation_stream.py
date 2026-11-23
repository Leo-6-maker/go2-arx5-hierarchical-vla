"""Causal two-camera, one-arm observation input for the frozen RoboDojo real policy."""

from collections import deque
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation


def rigid(value):
    frame = np.array(value, dtype=float, copy=True)
    if (
        frame.shape != (4, 4)
        or not np.isfinite(frame).all()
        or not np.allclose(frame[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
        or not np.allclose(
            frame[:3, :3].T @ frame[:3, :3], np.eye(3), rtol=0, atol=1e-8
        )
        or abs(np.linalg.det(frame[:3, :3]) - 1) > 1e-8
    ):
        raise ValueError("invalid measured or calibrated rigid transform")
    return frame


def camera(rgb):
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.shape != (256, 256, 3):
        raise ValueError("expected target camera 256x256 uint8 RGB")
    return a.copy()


class ObservationStream:
    """One instance per physical reset. Query only the latest simultaneous sensor frame."""

    def __init__(self, episode, prompt, virtual_left, source_origin, link6_tcp):
        if not episode or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("missing episode or task instruction")
        self.episode, self.prompt = episode, prompt
        self.left = np.array(virtual_left, dtype=float, copy=True)
        origin = np.asarray(source_origin, dtype=float)
        if (
            self.left.shape != (8,)
            or origin.shape != (3,)
            or not all(np.isfinite(a).all() for a in (origin, self.left))
        ):
            raise ValueError("invalid source calibration")
        self.base_origin = np.eye(4)
        self.base_origin[:3, 3] = origin
        self.link6_tcp = rigid(link6_tcp)
        self._lock = Lock()
        self._stopped = False
        self._ticks = deque()
        self._body = {}
        self._first = self._wrist = self._state = None

    def stop(self, episode):
        with self._lock:
            if episode != self.episode:
                raise ValueError("wrong stop epoch")
            self._stopped = True

    def observe(
        self, episode, tick, time_s, body, wrist, armbase, link6, tcp, finger_q_m
    ):
        with self._lock:
            if self._stopped:
                raise ValueError("episode stopped")
            if (
                episode != self.episode
                or type(tick) is not int
                or tick < 0
                or (not self._ticks and tick != 0)
                or (self._ticks and tick <= self._ticks[-1])
                or not np.isfinite(time_s)
                or abs(time_s - 0.002 * tick) > 1e-8
            ):
                raise ValueError("wrong reset or native observation clock")
            body, wrist = camera(body), camera(wrist)
            base, flange, tip = (rigid(f) for f in (armbase, link6, tcp))
            fingers = np.asarray(finger_q_m, float)
            if (
                fingers.shape != (2,)
                or not np.isfinite(fingers).all()
                or not np.allclose(flange @ self.link6_tcp, tip, rtol=0, atol=2e-9)
            ):
                raise ValueError("invalid actual finger state or TCP calibration")
            local = (
                np.linalg.inv(base @ self.base_origin)
                @ tip
                @ np.linalg.inv(self.link6_tcp)
            )
            wxyz = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
            if wxyz[0] < 0:
                wxyz = -wxyz
            self._state = np.r_[
                self.left, local[:3, 3], wxyz, fingers.mean() / 0.044
            ].astype(np.float32)
            if tick == 0:
                self._first = body
            self._wrist = wrist
            while self._ticks and self._ticks[0] < tick - 4000:
                del self._body[self._ticks.popleft()]
            self._ticks.append(tick)
            self._body[tick] = body

    def policy_input(self):
        with self._lock:
            if self._stopped:
                raise ValueError("episode stopped")
            if not self._ticks:
                raise ValueError("missing reset observation")
            if not 0 <= self._state[15] <= 1:
                raise ValueError("measured gripper outside source policy range")
            tick = self._ticks[-1]
            times = [max(0, tick - 500 * k) for k in range(8, 0, -1)] + [tick]
            try:
                head = np.stack(
                    [self._first]
                    + [self._first if t == 0 else self._body[t] for t in times]
                )
            except KeyError as e:
                raise ValueError(
                    f"missing required body observation at tick {e.args[0]}"
                ) from e
            return dict(
                images=dict(
                    cam_high=head,
                    cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                    cam_right_wrist=self._wrist.copy(),
                ),
                state=self._state.copy(),
                prompt=self.prompt,
            )
