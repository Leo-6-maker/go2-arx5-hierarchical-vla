"""Native four-parent DEV onset probe; requires new registered immutable source-only run."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'local-PC-native',a.output,physics=True)as spec:
        if spec.get('domain')!='S_CONTACT_DEV_ONSET_PROBE_V2'or spec.get('target_physics')is not False:
            raise PermissionError('Explicit separate DEV-only source identity required')
        from go2arx5_vla.world_model.contact_onset import run
        print(json.dumps(run(spec,a.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
