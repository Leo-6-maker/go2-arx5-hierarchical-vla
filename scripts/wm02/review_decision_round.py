"""Read and verify published WM02 round evidence without starting any workload."""
import argparse,hashlib,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args(argv)
    root=a.report_dir.resolve();manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,pin in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root)or hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('Evidence identity differs:'+name)
    result=json.loads((root/'SUMMARY.json').read_text())
    print(json.dumps(dict(evidence_hashes_verified=True,**result),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
