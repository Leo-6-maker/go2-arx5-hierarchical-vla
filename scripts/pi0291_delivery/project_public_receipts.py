"""Lossless identity projection of private receipts, not a dataset rewrite."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PRIVATE_PATH = re.compile(r'/(?:home|var/cache|scratch|tmp|mnt)/')


def project(value, key=''):
    assert key.lower() not in {'token','access_token','password','private_key','authorization_header'}, 'Authentication is not report input'
    if isinstance(value, dict):
        return {name:project(item,name) for name,item in value.items()}
    if isinstance(value, list):
        return [project(item,key) for item in value]
    if key == 'seed' and isinstance(value,int) and not isinstance(value,bool):
        return str(value)
    if isinstance(value,str) and PRIVATE_PATH.search(value):
        return 'PRIVATE_PATH_DETAIL_RETAINED_IN_OWNER_RECEIPT'
    return value


def publish(source, output, mappings):
    source, output = Path(source),Path(output)
    proof = []
    destinations = set()
    for mapping in mappings:
        name, destination = mapping.split('=',1)
        source_relative = Path(name)
        assert not source_relative.is_absolute() and '..' not in source_relative.parts
        relative = Path(destination)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert destination not in destinations
        destinations.add(destination)
        original = (source/name).read_bytes()
        if name.endswith('.jsonl'):
            values = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
            encoded = ''.join(json.dumps(project(row),allow_nan=False)+'\n' for row in values).encode()
        else:
            encoded = (json.dumps(project(json.loads(original)),indent=2,allow_nan=False)+'\n').encode()
        assert not PRIVATE_PATH.search(encoded.decode())
        target = output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(encoded)
        proof.append(dict(public_file=destination,producer_receipt_sha256=hashlib.sha256(original).hexdigest(),
            public_projection_sha256=hashlib.sha256(encoded).hexdigest(),
            original_private_receipt_retained=True))
    manifest = dict(status='SANITIZED_PROJECTION_NOT_PRODUCER_BYTES',files=proof,
        transformations=['Private absolute path-bearing values replaced with a placeholder',
            'Exact integer seed rendered as decimal string, never passed through floating point'],
        producer_HASH_and_public_HASH_are_distinct=True,
        dataset_mutation=False,physics_steps=0,model_loads=0)
    with (output/'PUBLIC_RECEIPT_PROJECTION_MANIFEST.json').open('x') as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=manifest['status'],files=len(proof))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--file',action='append',required=True)
    args = parser.parse_args()
    publish(args.input,args.output,args.file)
