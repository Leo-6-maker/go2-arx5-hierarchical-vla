"""Bounded read-only structure audit of four pinned demo files, never unpickle."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile

import h5py

from scripts.pi0294.source_demo_download import sha, write


def number(name):
    match = re.search(r'(?:demo_|traj_|episode_?)(\d+)', name)
    return int(match.group(1)) if match else 10**9


def attrs(group):
    result = {}
    for key, value in group.attrs.items():
        text = str(value)
        result[key] = text if len(text) <= 12000 else dict(bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest())
    return result


def schema(group, prefix='', limit=180):
    result = []
    def walk(g, stem):
        for key in g.keys():
            if len(result) >= limit:
                raise ValueError('HDF5_SCHEMA_NODE_CAP')
            link = g.get(key, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError('HDF5_EXTERNAL_OR_SOFT_LINK_NOT_ADMITTED')
            obj = g[key]; name = stem + '/' + key
            if isinstance(obj, h5py.Group):
                result.append(dict(path=name, type='group'))
                walk(obj, name)
            else:
                if obj.is_virtual or obj.external:
                    raise ValueError('HDF5_EXTERNAL_OR_VIRTUAL_DATASET_NOT_ADMITTED')
                result.append(dict(path=name, type='dataset', shape=list(obj.shape), dtype=str(obj.dtype)))
    walk(group, prefix)
    return result


def inspect_h5(file, kind):
    with h5py.File(file, 'r') as source:
        if kind == 'robotwin':
            return dict(kind=kind, root_keys=list(source), root_attrs=attrs(source),
                        direct_episode_schema=schema(source), pixels_decoded=0,
                        success_reexecuted=False, simulator_imported=False)
        bank = source['data'] if kind == 'mimicgen' else source
        names = sorted([name for name in bank.keys() if number(name) < 10**9], key=lambda name: (number(name), name))
        first = []
        for name in names[:3]:
            first.append(dict(id=name, attrs=attrs(bank[name]), schema=schema(bank[name])))
        return dict(kind=kind, recorded_episode_groups=len(names), first_three=first,
                    root_keys=list(source), root_attrs=attrs(source), data_attrs=attrs(bank),
                    success_reexecuted=False, pixels_decoded=0, simulator_imported=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();cpu=time.process_time();started=time.time()
    receipt=json.loads((root/'DOWNLOAD_RECEIPT.json').read_text(encoding='utf-8'))
    assert receipt['status']=='PASS_FOUR_FIXED_PAYLOAD_SHA256'
    rows=[]
    for row in receipt['files']:
        file=root/row['relative_local_path']
        assert file.resolve().is_relative_to(root) and sha(file)==row['expected_sha256']
        result=dict(repo=row['repo'],revision=row['revision'],path=row['path'],sha256=sha(file))
        if file.suffix in ('.h5','.hdf5'):
            result.update(inspect_h5(file,'mimicgen' if 'mimicgen' in row['repo'] else 'maniskill'))
        else:
            with zipfile.ZipFile(file) as archive:
                members=archive.infolist()
                if len(members)>512 or sum(x.file_size for x in members)>512*1024**2:
                    raise ValueError('ZIP_DECLARED_SIZE_OR_MEMBER_COUNT_CAP')
                for member in members:
                    path=PurePosixPath(member.filename)
                    if path.is_absolute() or '..' in path.parts or '\\' in member.filename or ':' in member.filename or stat.S_ISLNK(member.external_attr>>16):
                        raise ValueError('ZIP_UNSAFE_MEMBER_PATH_OR_SYMLINK')
                names=sorted([x.filename for x in members if x.filename.endswith(('.h5','.hdf5')) and number(x.filename)<10**9],key=lambda x:(number(x),x))
                inspected=[]
                for name in names[:3]:
                    item=archive.getinfo(name)
                    if item.file_size>64*1024**2:raise ValueError('ZIP_SELECTED_MEMBER_SIZE_CAP')
                    data=archive.read(name)
                    info=inspect_h5(io.BytesIO(data),'robotwin')
                    inspected.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),audit=info))
                result.update(kind='robotwin_zip',members=len(members),declared_uncompressed_bytes=sum(x.file_size for x in members),
                    member_suffix_counts=dict(Counter(PurePosixPath(x.filename).suffix for x in members)),
                    HDF5_members=len(names),first_three_members=inspected,member_name_inventory_sha256=hashlib.sha256(json.dumps(sorted(x.filename for x in members)).encode()).hexdigest(),
                    unpickle_calls=0,extracted_members=0)
        rows.append(result)
    value=dict(schema='FIXED_DEMO_SMALLPACK_READ_ONLY_SCHEMA_AUDIT_V1',status='PASS_BOUNDED_READ_ONLY_STRUCTURE_AUDIT',rows=rows,
        inspector_sha256=sha(__file__),download_receipt_sha256=sha(root/'DOWNLOAD_RECEIPT.json'),started_epoch=started,
        CPU_s=time.process_time()-cpu,wall_s=time.time()-started,new_physics=0,new_render=0,new_neural_queries=0,
        new_training_updates=0,unpickle_calls=0,source_mutations=0,selected='NONE',deployment='NONE')
    write(root/'STRUCTURE_AUDIT.json',value)
    print(json.dumps(dict(status=value['status'],files=len(rows),CPU_s=value['CPU_s'],rows=[{k:r[k] for k in ['kind','path']+(['recorded_episode_groups'] if 'recorded_episode_groups' in r else ['HDF5_members','member_suffix_counts'])} for r in rows])))


if __name__=='__main__':main()
