"""Wire only the sealed Container qualification cases into the existing N1 runtime."""

import argparse
from pathlib import Path
import shutil


PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
GUARD = "84ca8e1be60380ab24459f1357d18d8ca2a9b5049854c8def17b985b96c8ecf5"
NATIVE = "a93e2588ea9770a6eaa886b4c1b33b4d5c9b579ad2c5b8eb800de98e15bde993"
SCENES = {
    "EF5_Q_CONTAINER_01": "182d7c288411eeeda50bfc6eac1527d5033d9239128101e97fb0193f88fb0279",
    "EF5_Q_CONTAINER_02": "f7445e2d8db4eb5cbb56c2c8e79b4fa5510ce2612e07cd928e137c192ff11642",
}
WORLD = Path("src/go2arx5_vla/world_model")
PARENTS = {
    "multitask_adapter_v6.py": "23b9c50c22280e9a7d6a528f2aebe0bbed1a2c8462b3c0d69f105c1b5e4204d8",
    "multitask_scope_v6.py": "a2b903b558531ff5acb3195eca2b2528dc16e8c33df79795b4efa2323cdfae2f",
    "multitask_teacher_episode_v6.py": "c4d4e663114386f9d6f3a85dd2f9ed4545aaa64ae5f6edf3b3ead197d54546b0",
}


def native_observer(spec):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from go2arx5_vla.world_model.storage import digest
    from v64_container_teacher import QUALIFICATION

    identity = spec["panel_entry_id"]
    if identity not in QUALIFICATION:
        raise PermissionError("Only the two fixed Container QUAL layouts use this contact role")
    family, color, side = QUALIFICATION[identity]
    bound = read_pin(spec["v64_container_qualification"])
    guard = bound["observer_guard"]
    result = read_pin(guard)
    scene = spec["model_path"]
    library = spec["observer_library"]
    if (bound["plan"] != PLAN or bound["scene"] != dict(path=scene, sha256=SCENES[identity])
            or bound["id"] != identity or guard["sha256"] != GUARD
            or bound["candidate_observer"] != dict(path=library, sha256=NATIVE)
            or result["library_sha256"] != NATIVE or result["native_integrations"] != 0
            or spec["frozen_files"][scene] != SCENES[identity] or digest(scene) != SCENES[identity]
            or spec["frozen_files"][library] != NATIVE or digest(library) != NATIVE
            or spec["sim_only_role"] != "EXPERT" or spec["collection_split"] != "QUAL"
            or spec["expert_training_eligible"] or spec["max_ticks"] != 50703
            or spec["mobile_condition"]["family"] != family or spec["mobile_condition"]["mode"] != "M_FREE"
            or spec["task_instruction_id"] != f"PLACE-{color}-{side}"):
        raise PermissionError("Container observer, geometry, transport and qualification scope must agree")
    return NATIVE


EDITS = {
    "multitask_adapter_v6.py": [
        ("        return dict(environment, MFREE_V6_SELECTED_COLOR=self.task.color)\n",
         "        if self.spec.get('v64_container_qualification'):\n"
         "            from scripts.sla.v64_container_episode import bound_episode\n"
         "            container = bound_episode(self)\n"
         "            environment = dict(environment, WM02_V64_CONTAINER_CASE=container.identity,\n"
         "                               WM02_V64_CONTAINER_RELEASE=str(container.window.path))\n"
         "        return dict(environment, MFREE_V6_SELECTED_COLOR=self.task.color)\n"),
        ("        return TwoObjectState(model, self.task.color)\n",
         "        if self.spec.get('v64_container_qualification'):\n"
         "            from scripts.sla.v64_container_episode import bound_episode\n"
         "            return bound_episode(self).state_reader(model)\n"
         "        return TwoObjectState(model, self.task.color)\n"),
    ],
    "multitask_scope_v6.py": [
        ("    for key,d in (('observer_library',observer['observer_candidate']),\n",
         "    expected_observer = observer['observer_candidate']\n"
         "    if spec.get('v64_container_qualification'):\n"
         "        from scripts.sla.v64_container_runtime import native_observer\n"
         "        expected_observer = dict(library_sha256=native_observer(spec))\n"
         "    for key,d in (('observer_library',expected_observer),\n"),
        ("def episode(adapter):\n",
         "def episode(adapter):\n"
         "    if adapter.spec.get('v64_container_qualification'):\n"
         "        from scripts.sla.v64_container_episode import run\n"
         "        return run(adapter)\n"),
    ],
    "multitask_teacher_episode_v6.py": [
        ("def episode(adapter, *, placement_teacher=None):\n",
         "def episode(adapter, *, placement_teacher=None, container=None):\n"),
        ("    if adapter.spec['scope'] == RELEASE_SCOPE:\n",
         "    if container is not None:\n"
         "        teacher, score, reader = container.initialize(initial)\n"
         "    if adapter.spec['scope'] == RELEASE_SCOPE:\n"),
        ("    if placement_teacher is None and teacher.landing is not None:\n",
         "    if placement_teacher is None and container is None and teacher.landing is not None:\n"),
        ("                    row['consumed_mask'][atom]=True\n",
         "                    row['consumed_mask'][atom]=True\n"
         "                    if container is not None:\n"
         "                        container.after_atom(stream,reader,score,sample,witness,teacher.stage)\n"),
    ],
}


def install_runtime(source, destination):
    from go2arx5_vla.world_model.storage import digest

    for name, expected in PARENTS.items():
        if digest(source / WORLD / name) != expected:
            raise PermissionError(f"Unexpected original N1 runtime file: {name}")
    for name, replacements in EDITS.items():
        edited = (source / WORLD / name).read_text()
        for old, new in replacements:
            if edited.count(old) != 1:
                raise ValueError(f"Source edit no longer exact: {name}")
            edited = edited.replace(old, new, 1)
        compile(edited, name, "exec")
        target = destination / WORLD / name
        target.parent.mkdir(exist_ok=True, parents=True)
        with target.open("x") as stream:
            stream.write(edited)
    target = destination / "scripts/sla"
    target.mkdir(exist_ok=True, parents=True)
    for name in ("runtime", "episode", "release_contract", "terminal", "teacher"):
        filename = f"v64_container_{name}.py"
        with (target / filename).open("xb") as stream:
            stream.write(Path(__file__).with_name(filename).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    options = parser.parse_args()
    install_runtime(options.source, options.destination)
