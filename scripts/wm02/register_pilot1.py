"""Register a new local PILOT1 scope through the original owner mechanism."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.pilot1_scope import register
    print(json.dumps(register(args.binding)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
