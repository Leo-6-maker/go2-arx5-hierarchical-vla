"""Finite AUTO1 target-native WM fit and strict checkpoint restore; Actor remains0."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for key in('spec','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.runtime import admitted_run
    with admitted_run(a.spec,a.run_id,'wm02-auto1-target-neural',a.output,gpu=True)as spec:
        from go2arx5_vla.world_model.target_training import run
        print(json.dumps(run(spec,a.output)));return 0
if __name__=='__main__':raise SystemExit(main())
