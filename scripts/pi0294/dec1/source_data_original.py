"""Exact TRAIN126 and C1 materialization, using original fixed HF dependencies."""
from pathlib import Path
import re

from scripts.pi0292.data_consumer import extract_checked
from scripts.pi0292.durable import read, write
from scripts.pi0294.e1.scope import DATA
from scripts.pi0294.fd1 import sha
from scripts.pi0294.pv1.data_delivery import download_dataset
from .authority import R0

PROFILE='ORIGINAL126_C1'
REPO='leoyuer/pi0291-arx5-fixed-base'
V3='6aa94f4f40d791bedd6b4e75489e4ede6504d10d'
V2='6075aac0cdc44b8d252d7540efbac208701c30b3'
V1='c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea'
C0='8132a0f689b63592d821a28b9d027e4e5bcbf5b6'
C1='ce5faf453266689f7f889b8e452dd4f0a9c61200'
C1_MANIFEST='a6c029e3e07f264a08ab47a757eb899b752e350e7a93f1db2dd87c03b88f0052'
DEPENDENCY='4422a81d0ebcc461f77fafed0bbba090ee6682ac7e870dd407cc2c594d1823b8'
INPUTS={
    'plans/V3_TRANSFER.json':'32ec7bbc39c2ae47217433975d857674834731b212bb28d1ddc1baf33d59e543',
    'plans/C0_TRANSFER.json':'427e23594db01e7a8ec62e4ecddefd751c879825176745ce2f405241737fb28f',
    'plans/C1_TRANSFER.json':'cab2001ee8d1dfb95c495f468e5a78b5f6b40b27b008255b5261566d73b38134'}


def train_rows(manifest):
    if manifest.get('schema')!='PI0294_FROZEN_MULTITASK_NATIVE_V1':raise PermissionError('SOURCE126_ORIGINAL_SCHEMA')
    rows=[r for r in manifest['episodes'] if r['split']=='TRAIN']
    if (len(rows)!=126 or len({r['id'] for r in rows})!=126
            or {r['id'] for r in rows if r['origin']=='legacy'}!={f'T291{i:03d}' for i in range(64)}
            or len([r for r in rows if r['origin']=='new'])!=62):
        raise PermissionError('SOURCE126_EXACT_TRAIN_SUPPORT_NO_DEV_RAW')
    return rows


class FixedFiles:
    def __init__(self, root, spec, guard, *, get=download_dataset, cache=None):
        self.root,self.spec,self.guard,self.get=Path(root),spec,guard,get
        self.cache=cache;self.new_bytes=0
        if (spec['repo_id']!=REPO or spec['revision'] not in (V1,V2,V3,C0,C1)
                or not spec['prefix'].startswith(('PI0291/','PI0292/','PI0294/'))):
            raise PermissionError('SOURCE126_EXISTING_PRIVATE_FIXED_REVISIONS_ONLY')
        prefix=spec['prefix']+'/'
        if isinstance(spec['files'],list):
            self.files={r['path'][len(prefix):]:r for r in spec['files'] if r['path'].startswith(prefix)}
        else:
            self.files={n:dict(r,path=prefix+n) for n,r in spec['files'].items()}
            self.files['READY.json']=dict(path=prefix+'READY.json',sha256=spec['READY_sha256'])
        self.bytes=0;self.fetched=[]

    def fetch(self,name):
        if name not in self.files or name.startswith('/') or '..' in Path(name).parts:
            raise PermissionError('SOURCE126_ONLY_PRESEALED_DELIVERY_MEMBERS')
        if name in self.fetched:raise FileExistsError('SOURCE126_DO_NOT_FETCH_SAME_MEMBER_TWICE')
        row=self.files[name];self.guard(row.get('bytes',1024**2))
        if self.cache is None:
            path=Path(self.get(REPO,row['path'],revision=self.spec['revision'],local_dir=self.root,token=None));new=True
        else:path,new=self.cache.download(self,name,row,self.get)
        if (sha(path)!=row['sha256'] or path.stat().st_size!=row.get('bytes',path.stat().st_size)
                or name=='READY.json' and path.stat().st_size>1024**2):
            raise ValueError('SOURCE126_ACTUAL_FIXED_REVISION_MEMBER_CHANGED')
        self.bytes+=path.stat().st_size;self.fetched.append(name)
        if new:self.new_bytes+=path.stat().st_size
        return path

    def archive(self,name,proof,target):
        path=self.fetch(name)
        if path.stat().st_size!=proof['bytes'] or sha(path)!=proof['sha256']:
            raise ValueError('SOURCE126_ARCHIVE_INDEX_IDENTITY')
        self.extract(path,target,proof['member_sha256'])

    def extract(self,path,target,members):
        if self.cache is None:return extract_checked(path,target,members,self.guard)
        return self.cache.extract(path,target,members)


def correction(root, client, guard, previous=None):
    ready=read(client.fetch('READY.json'));members=read(client.fetch('MEMBERS.json'))
    spec=client.spec
    if (ready.get('private') is not True or ready['prefix']!=spec['prefix']
            or ready['files']!=spec['files'] or ready['statistics_sha256']!=R0
            or ready['manifest_sha256']!=spec['manifest_sha256'] or ready.get('READY_uploaded_last') is not True):
        raise PermissionError('SOURCE126_EXACT_PRIVATE_CORRECTION_READY')
    archive=client.fetch('correction.tar.gz');client.extract(archive,root,members)
    manifest=read(root/'DATASET_MANIFEST.json')
    if sha(root/'DATASET_MANIFEST.json')!=spec['manifest_sha256'] or sha(root/'statistics.json')!=R0:
        raise ValueError('SOURCE126_CORRECTION_MANIFEST_OR_R0_CHANGED')
    bindings={r['id']:str(root/'raw'/r['id']) for r in manifest['episodes']}
    if previous is not None:
        data,old_bindings,old_spec=previous;dep=read(root/'C0_DEPENDENCY.json')
        if (dep!=ready.get('C0_dependency') or dep['revision']!=old_spec['revision']
                or dep['manifest_sha256']!=sha(data/'DATASET_MANIFEST.json')):
            raise PermissionError('SOURCE126_C1_EXACT_C0_DEPENDENCY')
        old={r['id']:r for r in read(data/'DATASET_MANIFEST.json')['episodes']}
        if set(dep['reused_raw_episode_ids'])!=set(old):raise PermissionError('SOURCE126_ALL_C0_LINEAGES_RETAINED')
        for row in manifest['episodes']:
            if row['id'] in old:
                if row!=old[row['id']]:raise ValueError('SOURCE126_C1_REUSED_C0_ROW_CHANGED')
                bindings[row['id']]=old_bindings[row['id']]
    from scripts.pi0294.e1.correction import independent_read
    audit=independent_read(root,bindings)
    return bindings,audit


def consume(root, unused_receipt, inputs, guard, *, cache=None):
    r=Path(root);datasets=r/'datasets';datasets.mkdir();downloads=r/'download';downloads.mkdir()
    specs={name:read(r/'plans'/file) for name,file in (('v3','V3_TRANSFER.json'),('C0','C0_TRANSFER.json'),('C1','C1_TRANSFER.json'))}
    clients={name:FixedFiles(downloads/name,spec,guard,cache=cache) for name,spec in specs.items()}
    v3=clients['v3'];ready=read(v3.fetch('READY.json'));index=read(v3.fetch('ARCHIVE_MEMBER_MANIFEST.json'))
    dependency_path=v3.fetch('ORIGINAL_DEPENDENCY.json');dependency=read(dependency_path)
    if (v3.spec['revision']!=V3 or ready['version']!='v3.0' or ready['fixture'] is not False
            or ready['manifest_sha256']!=DATA or ready['statistics_sha256']!=R0
            or sha(dependency_path)!=DEPENDENCY
            or dependency!=ready['original_dependency'] or dependency['revision']!=V2):
        raise PermissionError('SOURCE126_FIXED_V3_READY_AND_ORIGINAL_DEPENDENCY')
    original=datasets/'original';v3.archive('native_dataset.tar.gz',index['native_dataset'],original)
    if sha(original/'DATASET_MANIFEST.json')!=DATA or sha(original/'statistics.json')!=R0:
        raise ValueError('SOURCE126_ORIGINAL_FROZEN_MANIFEST_OR_STATISTICS')
    manifest=read(original/'DATASET_MANIFEST.json');rows=train_rows(manifest)
    clients['v2']=FixedFiles(downloads/'v2',dependency,guard,cache=cache)
    early=dependency['original_dependency'];clients['v1']=FixedFiles(downloads/'v1',early,guard,cache=cache)
    indices={}
    for name in ('v1','v2'):
        client=clients[name];value=read(client.fetch('READY.json'))
        if value.get('private') is not True or value.get('READY_uploaded_last',True) is not True:
            raise PermissionError('SOURCE126_PRIVATE_ORIGINAL_READY_REQUIRED')
        indices[name]=read(client.fetch('ARCHIVE_MEMBER_MANIFEST.json'))
    bindings={};raw=r/'raw/original'
    for row in rows:
        guard();identifier=row['id']
        if row['origin']=='legacy':
            if not re.fullmatch('T291[0-9]{3}',identifier):raise PermissionError('SOURCE126_ONLY_ORIGINAL_TRAIN_LEGACY_IDS')
            key='v1' if int(identifier[-3:])<56 else 'v2';client=clients[key];proof=indices[key][identifier]
        else:client=v3;proof=index[identifier]
        client.archive('episodes/'+identifier+'.tar.gz',proof,raw)
        if sha(original/row['file'])!=row['sha256']:raise ValueError('SOURCE126_FROZEN_NUMERIC_CHANGED')
        folder=raw/identifier
        if sha(folder/'samples.json')!=row['samples_sha256']:raise ValueError('SOURCE126_ORIGINAL_RAW_SAMPLE_IDENTITY')
        bindings[identifier]=str(folder)
    write(r/'ORIGINAL_RAW_BINDINGS.private.json',bindings)
    c0=datasets/'C0';c1=datasets/'C1'
    old_bindings,audit0=correction(c0,clients['C0'],guard)
    c1_bindings,audit1=correction(c1,clients['C1'],guard,(c0,old_bindings,specs['C0']))
    if sha(c1/'DATASET_MANIFEST.json')!=C1_MANIFEST:raise ValueError('SOURCE126_EXACT_CURRENT_C1_NOT_OLDER_CORRECTION')
    write(r/'C1_RAW_BINDINGS.private.json',c1_bindings)
    from scripts.pi0294.mp1.dataset import SourceDataset
    from scripts.pi0294.pv1.export import batch_inventory
    source=SourceDataset(original,inputs['official_source'],inputs['cosmos'],for_optimization=True,
        raw_bindings=bindings,correction_root=c1,correction_bindings=c1_bindings)
    source.initialize();chosen={}
    for i,(row,anchor) in enumerate(source.rows):
        lane='original' if i<source.original_rows else 'C1'
        chosen.setdefault((lane,row['task'],row['family']),[]).append(i)
    probes=[];samples=[]
    for lane,indices_for_lane in chosen.items():
        for i in (indices_for_lane[0],indices_for_lane[len(indices_for_lane)//2],indices_for_lane[-1]):
            guard();sample=source[i]['processed'];batch=batch_inventory(source.processor.collator([sample]),1)
            probes.append(dict(lane=list(lane),identity=source.identity(i),batch=batch));samples.append(sample)
    mixed=batch_inventory(source.processor.collator([samples[0],samples[-1]]),2)
    for client in clients.values():
        if any('/PI0294-DEV-' in x or '/D291' in x for x in client.fetched):
            raise PermissionError('SOURCE126_NO_DEV_RAW_FETCH')
    return dict(status='PASS_VANDA_CPU_ORIGINAL126_AND_C1_FIXED_MEMBERS_AND_OFFICIAL_READER',
        original_manifest_sha256=sha(original/'DATASET_MANIFEST.json'),C1_manifest_sha256=sha(c1/'DATASET_MANIFEST.json'),
        statistics_sha256=R0,original_TRAIN_parents=126,C1_rows=len(read(c1/'DATASET_MANIFEST.json')['episodes']),
        C1_student_parents=len({x['source_student_episode'] for x in read(c1/'DATASET_MANIFEST.json')['episodes']}),
        frozen_native_metadata_and_numeric_DEV_retained_verbatim_not_used_for_optimization=True,DEV_raw_fetched=0,
        correction_independent_readers=dict(C0=audit0,C1=audit1),actual_CPU_official_batches=probes,mixed_CPU_batch=mixed,
        full_member_hashes_verified=True,all_original_TRAIN_raw_bindings_complete=True,
        new_download_bytes=sum(x.new_bytes for x in clients.values()),files_fetched={n:c.fetched for n,c in clients.items()},
        GPU=0,PBS=0,models=0,new_optimizer_updates=0,new_physics=0,training_qualification=False)
