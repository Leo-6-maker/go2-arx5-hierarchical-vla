"""Register the finite FUSION supplement through the existing owner mechanism."""
import argparse, json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.fusion_scope import register
    from scripts.pi0294.e1.journal import Journal
    from go2arx5_vla.world_model.storage import digest
    value = json.loads(args.binding.read_text())
    print(json.dumps(Journal(Path(value['campaign_root'])/'control/journal',
        Path(value['campaign_root']).name).once(Path(value['campaign_root']).name+'-FUSION-NIGHT1-REGISTER',
        dict(binding_sha256=digest(args.binding)), lambda: register(args.binding))))
    return 0


if __name__ == '__main__': raise SystemExit(main())
