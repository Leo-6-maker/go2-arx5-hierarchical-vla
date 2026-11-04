"""Render a private saved native recording with original MuJoCo meshes; no dynamics."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,choices=(15,30),default=30);p.add_argument('--max-wall',type=float,default=90)
    a=p.parse_args(argv)
    from go2arx5_vla.world_model.recording_video import render
    result=render(json.loads(a.config.read_text()),a.output,fps=a.fps,max_wall=a.max_wall);print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
