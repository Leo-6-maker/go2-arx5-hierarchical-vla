"""Open a private, SHA-bound saved recording; this entry cannot start a target executor."""
import argparse,json,subprocess
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-wall',type=float,default=300);p.add_argument('--rate',type=float,default=1.)
    a=p.parse_args(argv)
    if not 1<=a.max_wall<=600 or not .01<=a.rate<=4:p.error('Finite local recorded playback required')
    from go2arx5_vla.world_model.native_replay import passive
    config=json.loads(a.config.read_text());repo=Path(__file__).resolve().parents[2];q=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5);config['source_sha']=q.stdout.strip()if q.returncode==0 else None
    a.output.mkdir(exist_ok=False);print(json.dumps(passive(config,a.output,a.max_wall,a.rate)));return 0
if __name__=='__main__':raise SystemExit(main())
