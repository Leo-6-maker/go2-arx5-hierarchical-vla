"""Qualify one fixed prefix from actual TRAIN and already evaluated candidate."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError('KEEP_PREVIOUS_QUALIFICATION')
    from go2arx5_vla.world_model.stable_task_prefix_qualification_v1 import qualify
    from go2arx5_vla.world_model.storage import write_json
    v=json.loads(args.inputs.read_bytes())
    result=qualify(v['candidate'],v['TRAIN_native_result'],v['DEV_native_result'])
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],new_integrations=0,new_updates=0)))
    return 0


if __name__=='__main__':raise SystemExit(main())
