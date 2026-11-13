"""Build one native admission from actual closed3K and fresh-RGB artifacts.

The request names candidate_owner, training_result, complete_state,
parameter_inventory, ordinary_RGB_owner, ordinary_RGB_reconstruction and
ordinary_RGB_spec files, plus method and a new output file. This is metadata
and byte verification only; it starts no model, native process or scheduler.
"""
import argparse
import json
from pathlib import Path

from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.sla_dense_native_v5 import ADMISSION,NAMED_ADMISSION,NAMED_METHOD,parameter_bytes,qualification


def build(request):
    keys=('candidate_owner','training_result','complete_state','parameter_inventory',
          'ordinary_RGB_owner','ordinary_RGB_reconstruction','ordinary_RGB_spec')
    pins={key:dict(path=str(Path(request[key]).resolve(strict=True)),sha256=digest(request[key]))for key in keys}
    rs=json.loads(Path(pins['ordinary_RGB_spec']['path']).read_bytes())
    complete=json.loads(Path(pins['complete_state']['path']).read_bytes())
    inventory=json.loads(Path(pins['parameter_inventory']['path']).read_bytes())
    runtime={key:rs[key]for key in ('python','upstream_root','JAX_memory_fraction',
                                    'compilation_cache_dir','initial_function_rule')}
    q=dict(schema=NAMED_ADMISSION if request['method']==NAMED_METHOD else ADMISSION,
        method=request['method'],old_DEV_TEST_admitted=False,**pins,
        package=rs['package'],parent_params=rs['parent_params'],parent_manifest=rs['parent_manifest'],
        GPU_uuid=rs['GPU_uuid'],allocator_limit_bytes=rs['allocator_limit_bytes'],
        neural_owner_lock=rs['neural_owner_lock'],ordinary_parameter_bytes=parameter_bytes(inventory),
        dense_manifest_sha256=complete['bindings']['dense_manifest_sha256'],target_audit=rs['target_audit'],
        runtime=runtime,optimizer_updates=0,physical_requests=0,models_loaded=0)
    qualification(q)
    return q


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);args=p.parse_args()
    request=json.loads(args.request.read_bytes());output=Path(request['output'])
    if output.exists():raise FileExistsError('EXISTING_DENSE_NATIVE_ADMISSION_NEVER_OVERWRITE')
    q=build(request)
    with output.open('x')as f:json.dump(q,f,indent=2,allow_nan=False)
    print(json.dumps(dict(admission=dict(path=str(output),sha256=digest(output)),
        model_loads=0,optimizer_updates=0,physical_requests=0,native_task_success=False)))


if __name__=='__main__':main()
