"""Run one registered finite two-environment pilot; no persistent queue."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for name in('spec','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.target_parallel import run_batch
    result=run_batch(a.spec,a.output,a.run_id);print(json.dumps(result));return 0 if result['status']=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'else 1
if __name__=='__main__':raise SystemExit(main())
