"""Register NEXT2 through the actual original owner; no physics or CUDA."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.next2_scope import register
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(register(a.binding)));return 0
if __name__=='__main__':raise SystemExit(main())
