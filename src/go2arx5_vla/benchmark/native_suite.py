"""Deterministic strata selection for the Stage16 native benchmark."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "")).lower()
        for key in ("trajectory_id", "family", "original_class")
    )


def _choose(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    predicate,
    selected: set[str],
) -> list[Mapping[str, Any]]:
    pool = [
        row
        for row in rows
        if str(row.get("trajectory_id")) not in selected and predicate(row)
    ]
    pool.sort(
        key=lambda row: (
            str(row.get("family", "")),
            -float(row.get("descriptor", {}).get("path_length_m", 0.0)),
            str(row.get("trajectory_id", "")),
        )
    )
    return pool[:count]


def _append(
    output: list[dict[str, Any]], rows: Iterable[Mapping[str, Any]], cohort: str, selected: set[str]
) -> None:
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in selected:
            continue
        selected.add(trajectory_id)
        output.append(
            {
                "case_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "shape_id": trajectory_id,
                "placement_id": row.get("placement_profile", {}).get("profile_id"),
                "episode_binding_id": None,
                "cohort": cohort,
                "native_class": row.get("native_classification", {}).get("native_class"),
                "family": row.get("family"),
                "source": row.get("source"),
                "shape_file": row.get("shape_file"),
                "shape_sha256": row.get("shape_sha256"),
                "placement_profile": row.get("placement_profile", {}),
                "native_classification": row.get("native_classification", {}),
                "descriptor": row.get("descriptor", {}),
            }
        )


def build_native_suite(
    entries: Sequence[Mapping[str, Any]],
    progressive_ids: Iterable[str],
    *,
    target_count: int = 28,
    required_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build a 24--32 case suite without importing private runtime code."""

    if not 24 <= int(target_count) <= 32:
        raise ValueError("target_count must be between 24 and 32")
    rows = list(entries)
    selected: set[str] = set()
    output: list[dict[str, Any]] = []

    progressive_set = {str(value) for value in progressive_ids}
    # Reserve explicit gate cases before filling strata.  This keeps the
    # first-eight gate representative even when alphabetical/family ranking
    # would otherwise choose a different easy subset.
    required_rows = []
    by_id = {str(row.get("trajectory_id")): row for row in rows}
    for trajectory_id in required_ids:
        row = by_id.get(str(trajectory_id))
        if row is not None:
            required_rows.append(row)
    _append(
        output,
        required_rows,
        "PROGRESSIVE_BASE_RELOCATION",
        selected,
    )
    if required_rows:
        for row in output:
            if row["trajectory_id"] not in progressive_set:
                row["cohort"] = "NATIVE_GATE_REQUIRED"

    _append(
        output,
        [row for row in rows if str(row.get("trajectory_id")) in progressive_set],
        "PROGRESSIVE_BASE_RELOCATION",
        selected,
    )

    easy = [
        row
        for row in rows
        if row.get("native_classification", {}).get("native_class") == "NATIVE_ARM_EASY"
    ]
    boundary = [
        row
        for row in rows
        if row.get("native_classification", {}).get("native_class") == "NATIVE_BOUNDARY"
    ]
    if required_rows:
        # With the eight explicit gate cases reserved above, these allocations
        # produce a 28-case first benchmark when the source strata are large
        # enough: 8 progressive, 6 easy/gate, 4 orientation, 4 curvature,
        # 4 speed-controlled, and 2 boundary.  Short strata are filled below.
        _append(
            output,
            _choose(easy, count=1, predicate=lambda row: True, selected=selected),
            "NATIVE_ARM_EASY",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=3,
                predicate=lambda row: "orientation" in _text(row) or "coupling" in _text(row),
                selected=selected,
            ),
            "ORIENTATION_COMBINED_SE3",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=3,
                predicate=lambda row: "curvature" in _text(row) or "arc" in _text(row),
                selected=selected,
            ),
            "CURVATURE",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=4,
                predicate=lambda row: "speed" in _text(row) or any(
                    token in _text(row) for token in ("w2", "w3", "w4")
                ),
                selected=selected,
            ),
            "SPEED_CONTROLLED",
            selected,
        )
        _append(
            output,
            _choose(boundary, count=1, predicate=lambda row: True, selected=selected),
            "NATIVE_BOUNDARY",
            selected,
        )
    else:
        _append(
            output,
            _choose(easy, count=6, predicate=lambda row: True, selected=selected),
            "NATIVE_ARM_EASY",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=5,
                predicate=lambda row: "orientation" in _text(row) or "coupling" in _text(row),
                selected=selected,
            ),
            "ORIENTATION_COMBINED_SE3",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=4,
                predicate=lambda row: "curvature" in _text(row) or "arc" in _text(row),
                selected=selected,
            ),
            "CURVATURE",
            selected,
        )
        _append(
            output,
            _choose(
                easy,
                count=4,
                predicate=lambda row: "speed" in _text(row),
                selected=selected,
            ),
            "SPEED_CONTROLLED",
            selected,
        )
        _append(output, _choose(boundary, count=2, predicate=lambda row: True, selected=selected), "NATIVE_BOUNDARY", selected)

    # Fill only from the native easy/boundary pool if a requested stratum had
    # fewer unique rows than its nominal allocation.  Never fill with the
    # native-infeasible class or silently reuse the original E2 labels.
    _append(
        output,
        _choose(
            easy + boundary,
            count=max(0, int(target_count) - len(output)),
            predicate=lambda row: True,
            selected=selected,
        ),
        "NATIVE_SUITE_FILL",
        selected,
    )
    return output[: int(target_count)]
