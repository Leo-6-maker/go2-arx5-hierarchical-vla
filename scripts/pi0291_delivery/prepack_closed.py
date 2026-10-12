"""Prepackage stable completed records while the single physical owner continues."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import time

from archive_transfer import EPISODE_METADATA, contained_file, guard, read, sha, verified_tar, write
from deadline_guard import install_deadline


def main(args):
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]))
    root = Path(args.root)
    install_deadline(root,'prepack_'+args.receipt_name)
    receipt = root/'evidence'/args.receipt_name
    assert re.fullmatch(r'PREPACK_[A-Za-z0-9_]+\.json',args.receipt_name) and not receipt.exists()
    archive = root/'delivery'
    archive.mkdir(exist_ok=True)
    began = time.monotonic()
    own = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = own.ru_utime+own.ru_stime
    rows = []
    for identifier in args.ids:
        assert re.fullmatch(r'(?:[CP]291[0-9]{2}|[TD]291[0-9]{3})',identifier)
        folder = root/'episodes'/identifier
        result = read(folder/'RESULT.json')
        assert result['status']!='STARTED'
        original_manifest_sha = sha(folder/'FILE_MANIFEST.json')
        manifest = read(folder/'FILE_MANIFEST.json')
        assert all(item['path'] in EPISODE_METADATA or re.fullmatch(r'(?:body_rgb|wrist_rgb)/[0-9]{4}\.png',item['path']) for item in manifest)
        members = {identifier+'/'+item['path']:contained_file(folder,item['path']) for item in manifest}
        assert all(sha(folder/item['path'])==item['sha256'] for item in manifest)
        members[identifier+'/FILE_MANIFEST.json'] = folder/'FILE_MANIFEST.json'
        guard(root,cpu0,int(sum(path.stat().st_size for path in members.values())*1.1)+1024**2)
        proof = verified_tar(archive/(identifier+'.tar.gz'),members)
        assert sha(folder/'FILE_MANIFEST.json') == original_manifest_sha
        rows.append(dict(id=identifier,source_sha=result['source_sha'],raw_manifest_sha256=original_manifest_sha,
                         files=proof['files'],bytes=proof['bytes'],archive_sha256=proof['sha256']))
        print(json.dumps(dict(prepacked=identifier,bytes=proof['bytes'])),flush=True)
    own = resource.getrusage(resource.RUSAGE_SELF)
    value = dict(status='PASS_COMPLETED_RECORDS_PREPACKED_NO_UPLOAD',episodes=rows,
        CPU_s=own.ru_utime+own.ru_stime,elapsed_s=time.monotonic()-began,
        physical_steps=0,render_calls=0,HF_calls=0,delivery_code_sha256=sha(__file__))
    write(receipt,value)
    print(json.dumps(dict(status=value['status'],CPU_s=value['CPU_s'],episodes=len(rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--ids',nargs='+',required=True)
    parser.add_argument('--receipt-name',required=True)
    main(parser.parse_args())
