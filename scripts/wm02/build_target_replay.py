"""Build target-native numerical replay from admitted actual episodes; no physics."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    from go2arx5_vla.world_model.target_replay import build
    print(json.dumps(build(a.manifest,a.output)));return 0
if __name__=='__main__':raise SystemExit(main())
