"""Finite actual-allocation numerical diagnosis; zero optimizer updates."""
import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    spec = json.loads(parser.parse_args().spec.read_bytes())
    if (spec['operation'] != 'FIXED32_MICROBATCH_NUMERICS_NO_UPDATE_V3' or spec['microbatch'] != 1
            or spec['cohort_microbatch_points'] != [1, 8, 16] or spec.get('resume')):
        raise ValueError('FINITE_SAME_PARENT_NO_UPDATE_DIAGNOSIS_REQUIRED')
    from go2arx5_vla.sla_adapter.site_v2 import validate
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    from go2arx5_vla.sla_adapter.runtime_v3 import DeviceResidentEngine
    from go2arx5_vla.sla_adapter.numeric_diagnostic_v3 import diagnose
    site = validate(spec)
    root = Path(spec['registered_root'])
    if any((root / name).exists() for name in ('RESULT.private.json', 'FAILED.private.json', 'work')):
        raise FileExistsError('UNKNOWN_OR_FINISHED_DIAGNOSIS_NEVER_REPLAY')
    engine = DeviceResidentEngine.__new__(DeviceResidentEngine)
    began = time.time()
    try:
        DeviceResidentEngine.__init__(engine, spec, root / 'work')
        result = diagnose(engine, root, predecessor=read_pin(spec['predecessor_gradient']))
        result.update(status='DIAGNOSIS_COMPLETE_NOT_TRAINING_QUALIFICATION', exit_code=0,
                      source_sha=spec['source_sha'], site=site, wall_s=time.time() - began,
                      residency=engine.residency())
        with (root / 'RESULT.private.json').open('x') as stream:
            json.dump(result, stream, indent=2)
    except BaseException:
        with (root / 'FAILED.private.json').open('x') as stream:
            json.dump(dict(status='FAILED_DIAGNOSIS_PRESERVED', failure=traceback.format_exc(),
                           actual_engine_counts=getattr(engine, 'counts', {}),
                           optimization_visits=0, physical_requests=0,
                           wall_s=time.time() - began), stream, indent=2)
        raise
    finally:
        if hasattr(engine, 'loader'):
            engine.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
