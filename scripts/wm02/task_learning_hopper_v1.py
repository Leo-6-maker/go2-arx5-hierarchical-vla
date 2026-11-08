"""One finite fresh PBS allocation, original Trainer, no nested submission."""
import argparse
import json
import time
import traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.task_learning_hopper_v1 import run
    config=json.loads(args.config.read_text());root=Path(config['root'])
    code=1;result=None;failure=None;started=time.monotonic()
    try:
        result=run(config);code=0
    except BaseException:
        failure=traceback.format_exc();raise
    finally:
        write(root/'RESULT.private.json',result or dict(status='HOPPER_FAILURE_PRESERVED',failure=failure))
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.monotonic()-started,failure=failure))
        lease=json.loads((root/'control/LEASE.json').read_text())
        lease.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time(),
            result_sha256=digest(root/'RESULT.private.json'))
        write(root/'control/LEASE.json',lease)
    return code


if __name__=='__main__':raise SystemExit(main())
