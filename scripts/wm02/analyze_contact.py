"""Summarize frozen contact effects/coverage without refitting or native execution."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','predictions','result','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.contact_analysis import analyze
    result=analyze(a.replay,a.predictions,a.result)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'oracle_improvable_anchors':result['oracle']['improvable_vs_nominal'],
                      'grasp_coverage':result['contact_coverage'],'new_model_updates':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
