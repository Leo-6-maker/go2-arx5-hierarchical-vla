"""Register one actual Core-qualified DEV panel or one finite unexecuted group."""
import argparse,json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('panel','group','dispatch'))
    p.add_argument('--request',type=Path,required=True,
        help='Panel/group request JSON, or the already registered group INPUT for dispatch')
    a=p.parse_args(argv)
    from go2arx5_vla.world_model import task_learning_student_registration_v2 as registration
    if a.operation=='dispatch':result=registration.dispatch_group(a.request)
    else:result=getattr(registration,'register_'+a.operation)(json.loads(a.request.read_bytes()))
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
