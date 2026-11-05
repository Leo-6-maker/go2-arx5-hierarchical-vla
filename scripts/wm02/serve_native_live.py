"""Show the existing read-only native scene at a local browser URL."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--max-wall', type=float, default=3600.)
    parser.add_argument('--fps', type=float, default=12.)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or not 1 <= args.max_wall <= 86400 or not 1 <= args.fps <= 30:
        parser.error('finite endpoint 1..86400 seconds, 1..30 FPS, valid local port required')
    from go2arx5_vla.world_model.native_web import serve
    print(json.dumps(serve(json.loads(args.config.read_text()), args.output, port=args.port,
                           max_wall=args.max_wall, fps=args.fps)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
