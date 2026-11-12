"""Finite fixed-revision Hub download with complete byte verification.

This worker neither loads models nor creates training/physics admission. Run
only inside the existing registered owner/Journal path on the execution node.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.parse
import urllib.request


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def fetch(root, metadata, *, workers=2):
    repo, revision = metadata['id'], metadata['sha']
    if repo != 'haofuly/SLA_base' or len(revision) != 40 or workers not in (1, 2):
        raise ValueError('FIXED_SLA_BASE_METADATA_AND_BOUNDED_DOWNLOAD_REQUIRED')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def one(row):
        name = row['rfilename']
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('UNSAFE_HUB_MEMBER')
        target = root/name
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('ASSET_PATH_ESCAPE')
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = target.exists()
        if not reused:
            url = 'https://huggingface.co/'+repo+'/resolve/'+revision+'/'+urllib.parse.quote(name)
            temporary = target.with_name(target.name+'.partial')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as f:
                while block := response.read(4*1024**2):
                    f.write(block)
            if temporary.stat().st_size != row['size']:
                raise ValueError('DOWNLOADED_SIZE_MISMATCH:'+name)
            sha, git = hashes(temporary)
            if (row.get('lfs', {}).get('sha256', sha) != sha or
                    ('lfs' not in row and row['blobId'] != git)):
                raise ValueError('HUB_CONTENT_HASH_MISMATCH:'+name)
            temporary.rename(target)
        sha, git = hashes(target)
        if (target.stat().st_size != row['size'] or row.get('lfs', {}).get('sha256', sha) != sha or
                ('lfs' not in row and row['blobId'] != git)):
            raise ValueError('EXISTING_ASSET_MISMATCH_NO_OVERWRITE:'+name)
        result = dict(file=name, bytes=target.stat().st_size, sha256=sha, git_blob=git, reused=reused)
        print(json.dumps(dict(event='FILE_VERIFIED', **result)), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(one, metadata['siblings']))
    return dict(schema='SLA_BASE_BYTE_MANIFEST_V1', repo=repo, revision=revision,
                files=files, bytes=sum(f['bytes'] for f in files), all_files_verified=True,
                actual_model_loads=0, optimizer_updates=0, physical_requests=0, wall_s=time.time()-start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--result', required=True)
    a = p.parse_args()
    result = fetch(a.root, json.loads(Path(a.metadata).read_text()))
    with Path(a.result).open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
