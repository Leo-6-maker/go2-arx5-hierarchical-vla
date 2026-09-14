#!/usr/bin/env python3
"""PI0179 candidate replay recorder and dependency-light engineering smoke.

This module is an observational post-processor.  It consumes already-produced
camera frames, state/trajectory samples, and lifecycle telemetry; it never
starts a simulator, WBC, model, or robot process.  FFmpeg is used only after
the source frames and their timestamps have been durably described in JSON.

The input contract is intentionally explicit so a future closed-loop runner
can hand this recorder a candidate packet without changing the recorder's
semantics.  ``--synthetic-smoke`` creates a clearly marked visualization-only
fixture and is never a model-success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Color = tuple[int, int, int]
Point3 = tuple[float, float, float]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point3(value: Sequence[float], name: str) -> Point3:
    if len(value) != 3:
        raise ValueError(f"{name} must have length 3")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _distance(a: Point3, b: Point3) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ReplayFrameV1:
    """One source-timestamped observation for candidate replay."""

    timestamp_s: float
    body_frame: str
    wrist_frame: str
    achieved_eef_xyz: Point3
    target_xyz: Point3
    predicted_h40_xyz: tuple[Point3, ...]
    model_update_index: int
    safety_gate: str
    wbc_tracking_rms_position_m: float | None = None
    wbc_tracking_rms_orientation_rad: float | None = None
    lifecycle_state: str = "RUNNING"
    abort_reason: str | None = None
    safety_gate_issue_codes: tuple[str, ...] = ()
    target_distance_m: float | None = None
    initial_target_distance_m: float | None = None
    progress_m: float | None = None
    timeout: bool = False
    all_finite: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if not self.body_frame or not self.wrist_frame:
            raise ValueError("body_frame and wrist_frame are required")
        object.__setattr__(self, "achieved_eef_xyz", _point3(self.achieved_eef_xyz, "achieved_eef_xyz"))
        object.__setattr__(self, "target_xyz", _point3(self.target_xyz, "target_xyz"))
        points = tuple(_point3(point, "predicted_h40_xyz[]") for point in self.predicted_h40_xyz)
        if len(points) != 40:
            raise ValueError(f"predicted_h40_xyz must contain exactly 40 points, got {len(points)}")
        object.__setattr__(self, "predicted_h40_xyz", points)
        if int(self.model_update_index) < 0:
            raise ValueError("model_update_index must be non-negative")
        object.__setattr__(self, "model_update_index", int(self.model_update_index))
        if not self.safety_gate or not self.lifecycle_state:
            raise ValueError("safety_gate and lifecycle_state are required")
        for name in ("wbc_tracking_rms_position_m", "wbc_tracking_rms_orientation_rad", "target_distance_m", "initial_target_distance_m", "progress_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if not self.all_finite:
            raise ValueError("replay source contains non-finite values")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayFrameV1":
        return cls(
            timestamp_s=payload["timestamp_s"],
            body_frame=payload["body_frame"],
            wrist_frame=payload["wrist_frame"],
            achieved_eef_xyz=tuple(payload["achieved_eef_xyz"]),
            target_xyz=tuple(payload["target_xyz"]),
            predicted_h40_xyz=tuple(tuple(point) for point in payload["predicted_h40_xyz"]),
            model_update_index=payload["model_update_index"],
            safety_gate=payload["safety_gate"],
            wbc_tracking_rms_position_m=payload.get("wbc_tracking_rms_position_m"),
            wbc_tracking_rms_orientation_rad=payload.get("wbc_tracking_rms_orientation_rad"),
            lifecycle_state=payload.get("lifecycle_state", "RUNNING"),
            abort_reason=payload.get("abort_reason"),
            safety_gate_issue_codes=tuple(payload.get("safety_gate_issue_codes", ())),
            target_distance_m=payload.get("target_distance_m"),
            initial_target_distance_m=payload.get("initial_target_distance_m"),
            progress_m=payload.get("progress_m"),
            timeout=bool(payload.get("timeout", False)),
            all_finite=bool(payload.get("all_finite", True)),
        )

    def metric_row(self, initial_distance: float) -> dict[str, Any]:
        current_distance = _distance(self.achieved_eef_xyz, self.target_xyz)
        if self.target_distance_m is not None and abs(self.target_distance_m - current_distance) > 1e-6:
            raise ValueError("target_distance_m disagrees with source positions")
        initial = self.initial_target_distance_m if self.initial_target_distance_m is not None else initial_distance
        if abs(initial - initial_distance) > 1e-6:
            raise ValueError("initial_target_distance_m disagrees with first source frame")
        progress = initial - current_distance
        if self.progress_m is not None and abs(self.progress_m - progress) > 1e-6:
            raise ValueError("progress_m disagrees with source positions")
        return {
            "timestamp_s": self.timestamp_s,
            "target_distance_current_m": current_distance,
            "target_distance_initial_m": initial,
            "progress_m": progress,
            "safety_gate": self.safety_gate,
            "safety_gate_issue_codes": list(self.safety_gate_issue_codes),
            "wbc_tracking_rms_position_m": self.wbc_tracking_rms_position_m,
            "wbc_tracking_rms_orientation_rad": self.wbc_tracking_rms_orientation_rad,
            "model_update_index": self.model_update_index,
            "lifecycle_state": self.lifecycle_state,
            "abort_reason": self.abort_reason,
            "all_finite": self.all_finite,
            "timeout": self.timeout,
        }


@dataclass(frozen=True)
class ReplayRecorderConfigV1:
    output_dir: Path
    source_identity: Mapping[str, Any]
    fps: float = 20.0
    projection_scale_px_per_m: float = 240.0
    projection_origin_xy: tuple[int, int] = (160, 120)
    classification: str = "ENGINEERING_REPLAY_SMOKE"
    synthetic_fixture: bool = False
    ffmpeg_bin: str = "ffmpeg"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.source_identity:
            raise ValueError("source_identity is required")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError("fps must be positive and finite")
        if not math.isfinite(float(self.projection_scale_px_per_m)) or float(self.projection_scale_px_per_m) <= 0:
            raise ValueError("projection_scale_px_per_m must be positive and finite")
        if len(self.projection_origin_xy) != 2:
            raise ValueError("projection_origin_xy must have two values")


@dataclass
class _Image:
    width: int
    height: int
    pixels: bytearray

    def clone(self) -> "_Image":
        return _Image(self.width, self.height, bytearray(self.pixels))

    def _offset(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = self._offset(x, y)
            self.pixels[offset : offset + 3] = bytes(max(0, min(255, int(value))) for value in color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        for y in range(max(0, y0), min(self.height, y1)):
            for x in range(max(0, x0), min(self.width, x1)):
                self.set_pixel(x, y, color)

    def circle(self, cx: int, cy: int, radius: int, color: Color) -> None:
        radius = max(1, int(radius))
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                    self.set_pixel(x, y, color)

    def line(self, start: tuple[int, int], end: tuple[int, int], color: Color, width: int = 1) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for index in range(steps + 1):
            alpha = index / steps
            x = round(x0 + alpha * (x1 - x0))
            y = round(y0 + alpha * (y1 - y0))
            self.circle(x, y, max(0, width - 1), color)


def _read_ppm(path: Path) -> _Image:
    data = path.read_bytes()
    if not data.startswith((b"P6", b"P3")):
        raise ValueError(f"only PPM P6/P3 frames are supported without optional image libraries: {path}")
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            newline = data.find(b"\n", index)
            index = len(data) if newline < 0 else newline + 1
            continue
        end = index
        while end < len(data) and data[end] not in b" \t\r\n":
            end += 1
        if end == index:
            break
        tokens.append(data[index:end])
        index = end
    if len(tokens) != 4:
        raise ValueError(f"invalid PPM header: {path}")
    magic, width_raw, height_raw, max_raw = tokens
    width, height, maximum = int(width_raw), int(height_raw), int(max_raw)
    if width < 1 or height < 1 or maximum != 255:
        raise ValueError(f"unsupported PPM dimensions/max value: {path}")
    if magic == b"P6":
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        pixels = bytearray(data[index:])
    else:
        values = data[index:].split()
        pixels = bytearray(int(value) for value in values)
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"PPM pixel count mismatch for {path}: {len(pixels)} != {expected}")
    return _Image(width, height, pixels)


def _write_ppm(path: Path, image: _Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{image.width} {image.height}\n255\n".encode("ascii") + bytes(image.pixels))


_FONT: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"), "1": ("110", "010", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"), "I": ("111", "010", "010", "010", "111"),
    "L": ("100", "100", "100", "100", "111"), "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"), "P": ("110", "101", "110", "100", "100"),
    "R": ("110", "101", "110", "101", "101"), "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"), "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"), "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"), "Y": ("101", "101", "010", "010", "010"),
    "-": ("000", "000", "111", "000", "000"), ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"), "/": ("001", "001", "010", "100", "100"),
    "_": ("000", "000", "000", "000", "111"), " ": ("000", "000", "000", "000", "000"),
}


def _draw_text(image: _Image, x: int, y: int, text: str, color: Color = (240, 240, 240), scale: int = 2) -> None:
    cursor = x
    for char in text.upper():
        glyph = _FONT.get(char, _FONT[" "])
        for row, pattern in enumerate(glyph):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    image.fill_rect(cursor + column * scale, y + row * scale, cursor + (column + 1) * scale, y + (row + 1) * scale, color)
        cursor += 4 * scale


class ReplayRecorderV1:
    """Write replay videos and source-bound JSON sidecars without execution."""

    def __init__(self, config: ReplayRecorderConfigV1) -> None:
        self.config = config

    def _project(self, point: Point3, width: int, height: int) -> tuple[int, int]:
        origin_x, origin_y = self.config.projection_origin_xy
        # This is a declared overlay projection, not a replacement for native
        # camera calibration.  A future native adapter can bind pixel points.
        x = int(round(origin_x + point[1] * self.config.projection_scale_px_per_m))
        y = int(round(origin_y - point[2] * self.config.projection_scale_px_per_m))
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def _annotate(self, source: _Image, frame: ReplayFrameV1, history: Sequence[ReplayFrameV1]) -> _Image:
        image = source.clone()
        target = self._project(frame.target_xyz, image.width, image.height)
        achieved = self._project(frame.achieved_eef_xyz, image.width, image.height)
        image.circle(*target, 7, (235, 25, 25))
        image.circle(*achieved, 5, (30, 235, 70))
        previous = None
        for historical in history:
            point = self._project(historical.achieved_eef_xyz, image.width, image.height)
            if previous is not None:
                image.line(previous, point, (40, 220, 70), 2)
            previous = point
        prediction_previous = achieved
        for predicted in frame.predicted_h40_xyz:
            point = self._project(predicted, image.width, image.height)
            image.line(prediction_previous, point, (35, 215, 235), 1)
            prediction_previous = point
        panel_height = 45
        image.fill_rect(0, 0, image.width, panel_height, (8, 12, 18))
        initial = frame.initial_target_distance_m
        distance = _distance(frame.achieved_eef_xyz, frame.target_xyz)
        progress = (initial - distance) if initial is not None else 0.0
        lines = (
            f"DIST {distance:.3f} PROG {progress:.3f}",
            f"SAFE {frame.safety_gate} WBC {frame.wbc_tracking_rms_position_m if frame.wbc_tracking_rms_position_m is not None else -1.0:.3f}",
            f"UPD {frame.model_update_index} LIFE {frame.lifecycle_state}",
        )
        for index, line in enumerate(lines):
            _draw_text(image, 4, 2 + index * 14, line[: max(1, image.width // 8)], scale=1)
        return image

    @staticmethod
    def _encode(ffmpeg_bin: str, frame_dir: Path, output: Path, fps: float) -> None:
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"FFmpeg executable is unavailable: {ffmpeg_bin}")
        command = [
            ffmpeg_bin, "-y", "-loglevel", "error", "-framerate", f"{fps:.9g}",
            "-i", str(frame_dir / "%06d.ppm"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ]
        subprocess.run(command, check=True)

    def render(self, frames: Sequence[ReplayFrameV1]) -> dict[str, Any]:
        if not frames:
            raise ValueError("at least one replay frame is required")
        timestamps = [frame.timestamp_s for frame in frames]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("source timestamps must be strictly increasing")
        initial_distance = _distance(frames[0].achieved_eef_xyz, frames[0].target_xyz)
        metrics = [frame.metric_row(initial_distance) for frame in frames]
        output = self.config.output_dir.expanduser()
        if output.exists() and any(output.iterdir()) and not self.config.overwrite:
            raise FileExistsError(f"output directory is non-empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="pi0179_replay_", dir=str(output)))
        frame_dirs = {name: temp_root / name for name in ("third_person", "body_camera", "wrist_camera", "side_by_side")}
        for directory in frame_dirs.values():
            directory.mkdir()
        first_shape: tuple[int, int] | None = None
        overlay_rows: list[dict[str, Any]] = []
        try:
            for index, frame in enumerate(frames):
                body = _read_ppm(Path(frame.body_frame))
                wrist = _read_ppm(Path(frame.wrist_frame))
                if (body.width, body.height) != (wrist.width, wrist.height):
                    raise ValueError("body and wrist source frames must share dimensions")
                if first_shape is None:
                    first_shape = (body.width, body.height)
                elif first_shape != (body.width, body.height):
                    raise ValueError("source frame dimensions changed during replay")
                history = frames[max(0, index - 19) : index + 1]
                third = self._annotate(body, frame, history)
                wrist_overlay = self._annotate(wrist, frame, history)
                metrics_panel = _Image(body.width, body.height, bytearray([12, 16, 22] * (body.width * body.height)))
                metric = metrics[index]
                panel_lines = (
                    "PI0179 REPLAY",
                    f"DIST {metric['target_distance_current_m']:.3f} INIT {metric['target_distance_initial_m']:.3f}",
                    f"PROG {metric['progress_m']:.3f} SAFE {metric['safety_gate']}",
                    f"WBC {metric['wbc_tracking_rms_position_m'] if metric['wbc_tracking_rms_position_m'] is not None else -1.0:.3f}",
                    f"UPD {metric['model_update_index']} LIFE {metric['lifecycle_state']}",
                    f"ABORT {metric['abort_reason'] or 'NONE'}",
                )
                for line_index, line in enumerate(panel_lines):
                    _draw_text(metrics_panel, 6, 8 + line_index * 16, line[: max(1, body.width // 8)], scale=2)
                side = _Image(body.width * 2, body.height * 2, bytearray([0, 0, 0] * (body.width * 2 * body.height * 2)))
                for source, x0, y0 in ((third, 0, 0), (wrist_overlay, body.width, 0), (body, 0, body.height), (metrics_panel, body.width, body.height)):
                    for y in range(source.height):
                        start = ((y0 + y) * side.width + x0) * 3
                        side.pixels[start : start + source.width * 3] = source.pixels[y * source.width * 3 : (y + 1) * source.width * 3]
                for name, image in (("third_person", third), ("body_camera", body), ("wrist_camera", wrist), ("side_by_side", side)):
                    _write_ppm(frame_dirs[name] / f"{index:06d}.ppm", image)
                overlay_rows.append({
                    "frame_index": index,
                    "timestamp_s": frame.timestamp_s,
                    "target_xyz": list(frame.target_xyz),
                    "achieved_eef_xyz": list(frame.achieved_eef_xyz),
                    "achieved_eef_trail_xyz": [list(item.achieved_eef_xyz) for item in history],
                    "predicted_h40_xyz": [list(point) for point in frame.predicted_h40_xyz],
                    "model_update_index": frame.model_update_index,
                    "projection": {
                        "name": "PI0179_DECLARED_OVERLAY_PROJECTION_V1",
                        "status": "VISUALIZATION_ONLY_UNTIL_NATIVE_CAMERA_BINDING",
                        "origin_xy": list(self.config.projection_origin_xy),
                        "scale_px_per_m": self.config.projection_scale_px_per_m,
                    },
                })
            video_paths: dict[str, Path] = {}
            for name in frame_dirs:
                video_path = output / f"{name}.mp4"
                self._encode(self.config.ffmpeg_bin, frame_dirs[name], video_path, self.config.fps)
                video_paths[name] = video_path
            overlay_path = output / "trajectory_overlay.json"
            overlay_path.write_text(json.dumps({
                "schema": "STAGE45F_PI0179_TRAJECTORY_OVERLAY_V1",
                "source_timestamps_authoritative": True,
                "frames": overlay_rows,
            }, indent=2) + "\n", encoding="utf-8")
            metrics_path = output / "frame_metrics.jsonl"
            with metrics_path.open("w", encoding="utf-8") as handle:
                for row in metrics:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            manifest = {
                "schema": "STAGE45F_PI0179_REPLAY_MANIFEST_V1",
                "classification": self.config.classification,
                "source_identity": dict(self.config.source_identity),
                "source_timestamps_authoritative": True,
                "ffmpeg_postprocessing_only": True,
                "synthetic_fixture": self.config.synthetic_fixture,
                "simulator_executed": False,
                "wbc_executed": False,
                "hardware_executed": False,
                "grasp_contact_physics": False,
                "camera_layout": "third-person | wrist / body | metrics",
                "required_outputs": {
                    "third_person": "third_person.mp4",
                    "body_camera": "body_camera.mp4",
                    "wrist_camera": "wrist_camera.mp4",
                    "side_by_side": "side_by_side.mp4",
                    "trajectory_overlay": "trajectory_overlay.json",
                    "frame_metrics": "frame_metrics.jsonl",
                },
                "frame_count": len(frames),
                "source_fps_declared": self.config.fps,
                "source_frame_shape_hw": list(first_shape or (0, 0)),
                "files": {
                    **{name: {"path": path.name, "sha256": _sha256(path)} for name, path in video_paths.items()},
                    "trajectory_overlay": {"path": overlay_path.name, "sha256": _sha256(overlay_path)},
                    "frame_metrics": {"path": metrics_path.name, "sha256": _sha256(metrics_path)},
                },
                "source_frame_manifest_sha256": _canonical_sha([
                    {"timestamp_s": frame.timestamp_s, "body_frame": frame.body_frame, "wrist_frame": frame.wrist_frame}
                    for frame in frames
                ]),
            }
            manifest_path = output / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return manifest
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


def _synthetic_frames(source_dir: Path, count: int = 12) -> tuple[ReplayFrameV1, ...]:
    """Create a non-simulator renderer fixture; never a scientific result."""

    source_dir.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    target = (0.08, 0.06, 0.12)
    frames: list[ReplayFrameV1] = []
    initial = (-0.06, -0.04, 0.02)
    for index in range(count):
        alpha = index / max(1, count - 1)
        achieved = tuple(initial[axis] + alpha * (target[axis] - initial[axis]) for axis in range(3))
        body = _Image(width, height, bytearray([28, 36, 48] * (width * height)))
        wrist = _Image(width, height, bytearray([38, 45, 34] * (width * height)))
        # Fixture-only robot/arm silhouette and visible target.  These pixels
        # are not a simulator output and the manifest says so explicitly.
        for image in (body, wrist):
            image.line((35, 205), (110, 150), (120, 125, 135), 5)
            image.line((110, 150), (165, 125), (150, 150, 165), 5)
            image.circle(35, 205, 10, (90, 95, 105))
            image.circle(165, 125, 7, (80, 210, 90))
            image.circle(250, 75, 9, (235, 20, 20))
        body_path = source_dir / f"body_{index:06d}.ppm"
        wrist_path = source_dir / f"wrist_{index:06d}.ppm"
        _write_ppm(body_path, body)
        _write_ppm(wrist_path, wrist)
        predicted = tuple(
            tuple(achieved[axis] + (target[axis] - achieved[axis]) * (step + 1) / 40.0 for axis in range(3))
            for step in range(40)
        )
        frames.append(ReplayFrameV1(
            timestamp_s=index / 20.0,
            body_frame=str(body_path),
            wrist_frame=str(wrist_path),
            achieved_eef_xyz=achieved,
            target_xyz=target,
            predicted_h40_xyz=predicted,
            model_update_index=index,
            safety_gate="PASS",
            wbc_tracking_rms_position_m=0.002 + index * 0.0001,
            wbc_tracking_rms_orientation_rad=0.004,
            lifecycle_state="RUNNING" if index < count - 1 else "COMPLETE",
            initial_target_distance_m=_distance(initial, target),
        ))
    return tuple(frames)


def _load_frames(path: Path) -> tuple[ReplayFrameV1, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["frames"] if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("frames JSON must be a list or an object with a frames list")
    return tuple(ReplayFrameV1.from_dict(row) for row in rows)


def plan() -> dict[str, Any]:
    return {
        "schema": "STAGE45F_PI0179_REPLAY_RECORDER_PLAN_V1",
        "execution": False,
        "simulator_executed": False,
        "wbc_executed": False,
        "hardware_executed": False,
        "required_outputs": [
            "third_person.mp4", "body_camera.mp4", "wrist_camera.mp4", "side_by_side.mp4",
            "trajectory_overlay.json", "frame_metrics.jsonl", "manifest.json",
        ],
        "third_person_semantics": [
            "robot_and_arm", "red_target", "achieved_eef", "achieved_eef_trail",
            "predicted_h40", "receding_horizon_update_index",
        ],
        "telemetry": [
            "target_distance_current_and_initial", "progress", "safety_gate",
            "wbc_tracking_rms", "lifecycle", "abort_reason",
        ],
        "ffmpeg_role": "post-processing-only",
        "smoke_classification": "ENGINEERING_REPLAY_SMOKE",
        "grasp_contact_physics": "design-note-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--frames-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pi0179_replay_smoke"))
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan(), indent=2))
        return
    if args.synthetic_smoke == bool(args.frames_json):
        parser.error("choose exactly one of --synthetic-smoke or --frames-json")
    if args.synthetic_smoke:
        source_dir = args.output_dir / "_synthetic_source"
        frames = _synthetic_frames(source_dir)
        source_identity = {
            "kind": "SYNTHETIC_RENDERER_FIXTURE",
            "fixture": "PI0179_DEPENDENCY_FREE_REACH_VISUAL_V1",
            "scientific_claim": "NONE",
        }
        synthetic = True
    else:
        frames = _load_frames(args.frames_json)
        source_identity = {"kind": "EXTERNAL_SOURCE_FRAME_BUNDLE", "frames_json": str(args.frames_json)}
        synthetic = False
    result = ReplayRecorderV1(ReplayRecorderConfigV1(
        output_dir=args.output_dir,
        source_identity=source_identity,
        synthetic_fixture=synthetic,
        classification="ENGINEERING_REPLAY_SMOKE" if synthetic else "CANDIDATE_REPLAY_POSTPROCESSING_READY",
        ffmpeg_bin=args.ffmpeg_bin,
        overwrite=args.overwrite,
    )).render(frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
