"""Project original saved startup ticks only; no MuJoCo model/controller construction."""
import argparse,json,subprocess
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Preserve prior projection; choose a new output')
    m=json.loads(a.manifest.read_text());q=subprocess.run(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
    if q.returncode:raise RuntimeError('Current source SHA required for new projection')
    m['source_sha']=q.stdout.strip()
    from go2arx5_vla.world_model.startup_timeline import run
    r=run(m,a.output);print(json.dumps({k:v for k,v in r.items()if k!='episodes'},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
