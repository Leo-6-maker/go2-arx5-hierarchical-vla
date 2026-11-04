"""Verify/restore exact research commit objects after API publication; never update refs."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--write',action='store_true');a=p.parse_args(argv)
    v=json.loads(a.manifest.read_text())
    if v['schema']!='WM02_EXACT_SOURCE_PUBLICATION_MAP_V1':raise ValueError('Exact source map required')
    for row in v['snapshots']:
        raw=row['original_commit_object'].encode();pin=hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if pin!=row['actual_source_sha']or not raw.startswith(('tree '+row['tree_sha']+'\n').encode()):raise ValueError('Original source commit bytes differ')
        # Clone/fetch the published snapshot history first; its exact trees/blobs
        # must already exist. Metadata recovery is not a source download.
        subprocess.run(['git','-C',str(a.repo),'cat-file','-e',row['tree_sha']+'^{tree}'],check=True,capture_output=True)
        if a.write:
            q=subprocess.run(['git','-C',str(a.repo),'hash-object','-w','-t','commit','--stdin'],input=raw,check=True,capture_output=True)
            if q.stdout.decode().strip()!=pin:raise ValueError('Restored source identity differs')
    print(json.dumps(dict(status='EXACT_SOURCE_COMMIT_OBJECTS_RESTORED'if a.write else'EXACT_SOURCE_COMMIT_OBJECTS_VERIFIED',commits=len(v['snapshots']),ref_updates=0,checkouts=0,new_physics=0)));return 0
if __name__=='__main__':raise SystemExit(main())
