"""Observe an existing native collection run; never starts physics or controls."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall', type=float, default=300.)
    parser.add_argument('--fps', type=float, default=15.)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.max_wall <= 86400 or not 1 <= args.fps <= 30:
        parser.error('finite endpoint 1..86400 seconds; display rate 1..30 FPS')
    from go2arx5_vla.world_model.native_live import watch
    print(json.dumps(watch(json.loads(args.config.read_text()), args.output,
                           max_wall=args.max_wall, fps=args.fps, headless=args.headless)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
