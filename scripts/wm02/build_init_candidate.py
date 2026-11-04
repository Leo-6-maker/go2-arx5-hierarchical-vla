"""Compile/link and verify the new target candidate; never launch native simulator."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'core-wm02-init-build',a.output)as spec:
        if spec.get('domain')!='TARGET_INIT_BUILD_ONLY_V1'or spec.get('physics')is not False:raise PermissionError('Build-only identity required')
        from go2arx5_vla.world_model.init_candidate import build
        result=build(spec,a.output);print(json.dumps({k:v for k,v in result.items()if k!='actual_preload_chains'}))
    return 0
if __name__=='__main__':raise SystemExit(main())
