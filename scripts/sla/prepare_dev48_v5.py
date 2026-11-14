"""Compile the fixed new DEV48 layouts using original FK/contact checks."""
import json
from pathlib import Path
import sys

from go2arx5_vla.world_model import sla_dev48_v5 as dev
from go2arx5_vla.world_model.stable_task_scene_v1 import pinned
from go2arx5_vla.world_model.sla_population_v1 import geometry_sha
from go2arx5_vla.world_model.task_learning_scene_v1 import geometry
from go2arx5_vla.world_model.storage import write_json
from scripts.sla.prepare_static_v1 import prepare_population


def run(spec):
    d,p=dev.declaration()
    if (spec['operation']!=dev.STATIC_OPERATION or spec['preregistration']['sha256']!=p
            or pinned(spec['preregistration'])!=d):
        raise ValueError('FIXED_NEW_DEV48_STATIC_INPUTS_ONLY_NO_MODEL_OR_DYNAMIC_GRANT')
    rows=[dict(r,initial_geometry_sha256=geometry_sha(geometry(r['layout'])))for r in dev.inputs()]
    return prepare_population(spec,expected_schema=dev.DECLARATION_SCHEMA,expected_conditions=48,
        offsets=dev.OFFSETS,split=dev.SPLIT,result_schema=dev.STATIC_SCHEMA,
        disjoint_schema=dev.DISJOINT_SCHEMA,declared_population=rows)


if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    write_json(Path(spec['registered_root'])/'RESULT.private.json',run(spec))
