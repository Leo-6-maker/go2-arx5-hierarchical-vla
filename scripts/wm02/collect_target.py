"""One finite, admitted target sequential collection attempt (no Actor)."""
def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    for key in('spec','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    spec=json.loads(a.spec.read_text())
    if not spec['phase'].startswith('collect-target-'):raise PermissionError('REGISTERED_FINITE_COLLECTION_PHASE_REQUIRED')
    from .commission_target import main as run
    return run(['--spec',str(a.spec),'--output',str(a.output),'--run-id',a.run_id])
if __name__=='__main__':raise SystemExit(main())
