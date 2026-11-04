"""Finite source-only real model-before-action selector; no Actor optimizer or target nodes."""
import argparse,json
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-native',a.output,physics=True)as spec:
        if spec.get('domain')!='S_FROZEN_ONLINE_SELECTOR_V3'or spec.get('target_physics')is not False:
            raise PermissionError('Explicit finite source-only frozen selector required')
        from go2arx5_vla.world_model.online_selector import run
        print(json.dumps(run(spec,a.output)))
    return 0
if __name__=='__main__':raise SystemExit(main())
