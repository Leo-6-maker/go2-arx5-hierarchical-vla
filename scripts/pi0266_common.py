"""PI0266 paths and pinned source identity; no model or simulator imports."""
import json
from pathlib import Path
from pi0265_source import sha, write

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0266')
BASE = ROOT.parent
OLD = BASE/'PI0265'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0266')
DATA = RAW.parent/'PI0260/R2'
REF = BASE/'PI0250/runtime/reference-317a'
AUTHORITY = '5f728de34c900523fa933a75c82be1b310ad814b'


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources():
    source = read(OLD/'evidence/SOURCE_IDENTITY.json')
    for row in source['files']:
        assert sha(row['path']) == row['sha256'], row['name']
    seal = read(OLD/'design/EXECUTION_SEAL.json')
    for rel, digest in seal['files'].items():
        assert sha(OLD/rel) == digest, rel
    return source
