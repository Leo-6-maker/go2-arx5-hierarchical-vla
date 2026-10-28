"""Actual output/inode guard for the exact already-paid third D25 panel.

Metadata reads only. Original model SHA verification still belongs to loader;
this guard preserves the independent saved stream identity between effects.
"""
import hashlib,json,os,stat,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import mix_core_d25_local_successor as s
from .mix_core_context import LOCAL_BINDING

BINDING='plans/V5_CD_D25_FINITE_LOCAL_SUCCESSOR.private.json'

def need(ok,why):
    if not ok:raise PermissionError('D25_ACTUAL_PAID_OUTPUT_'+why)

def inventory(root,*,uid,maximum=100000):
    root=Path(root);need(root.is_absolute()and not root.is_symlink()and root.is_dir(),'OWN_REAL_ROOT')
    rows={};allocated={};stack=[root]
    while stack:
        p=stack.pop();v=p.lstat();need(v.st_uid==uid,'OWN_INODE_NOT_FOREIGN_OUTPUT')
        need(stat.S_ISDIR(v.st_mode)or stat.S_ISREG(v.st_mode)or stat.S_ISLNK(v.st_mode),'REGULAR_DIRECTORY_OR_PINNED_INPUT')
        if stat.S_ISDIR(v.st_mode):stack.extend(p.iterdir())
        rows[str(p)]=dict(type='symlink'if stat.S_ISLNK(v.st_mode)else'directory'if stat.S_ISDIR(v.st_mode)else'file',
            dev=v.st_dev,ino=v.st_ino,nlink=v.st_nlink,uid=v.st_uid,mode=stat.S_IMODE(v.st_mode),
            bytes=v.st_size,blocks=v.st_blocks,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns)
        if p.is_symlink():
            t=p.stat();rows[str(p)].update(link_text=os.readlink(p),canonical_target=str(p.resolve(strict=True)),
                target_stat=dict(dev=t.st_dev,ino=t.st_ino,bytes=t.st_size,mtime_ns=t.st_mtime_ns,ctime_ns=t.st_ctime_ns,nlink=t.st_nlink))
        allocated[(v.st_dev,v.st_ino)]=max(v.st_size,v.st_blocks*512)
        need(len(rows)<=maximum,'FINITE_OWN_METADATA_INVENTORY')
    return rows,allocated

class Guard:
    def __init__(self,root,manifest,request,local,*,clock=time.monotonic):
        self.root=Path(root);self.clock=clock;self.last=None;self.proof=None
        marker=s.validated(local);need(marker is not None and request['request_id']==marker['replacement_request_id'],'EXACT_ADOPTED_REQUEST')
        need(sha(self.root/BINDING)==manifest['execution']['sealed_inputs'].get(BINDING)==s.digest(marker)
             and read(self.root/BINDING)==marker,'IMMUTABLE_COMPLETE_PAID_IO_MARKER')
        self.marker=marker;self.join=marker['physical_inventory_join']
        binding=read(self.root/LOCAL_BINDING);self.owner=Path(binding['owner_root'])
        self.physical=self.root.parent.parent/'children'/self.root.name
        need(self.root.parent.name=='neural-children'and self.root.name==s.RUN
             and self.owner.parent==self.root.parent.parent/'control','ONLY_EXACT_REGISTERED_NEW_PEERS')
        self.uid=self.root.stat().st_uid;need(self.uid==os.getuid()==2009,'ACTUAL_CORE_ACCOUNT')

    def check(self,*,force=False):
        if not force and self.last is not None and self.clock()-self.last<30:return self.proof
        retained={}
        for item in self.join['roots'].values():
            old,_=inventory(item['path'],uid=self.uid);retained.update(old)
        original=(json.dumps(retained,sort_keys=True,separators=(',',':'))+'\n').encode()
        need(hashlib.sha256(original).hexdigest()==self.join['Core_inventory_sha256'],
             'ALL_CLOSED_PREDECESSOR_INODES_AND_OUTPUTS_RETAINED_UNCHANGED')
        model=Path(self.join['model_root']);model_used={}
        for name,witness in self.join['model_stats'].items():
            p=model/name;v=p.lstat();expected=witness['stat']
            need(stat.S_ISREG(v.st_mode)and v.st_uid==self.uid and stat.S_IMODE(v.st_mode)==0o400
                 and all(actual==expected[key]for key,actual in [('dev',v.st_dev),('ino',v.st_ino),('nlink',v.st_nlink),
                    ('bytes',v.st_size),('mtime_ns',v.st_mtime_ns),('ctime_ns',v.st_ctime_ns)]),
                 'UNCHANGED_INDEPENDENT_STREAM_WITNESS:'+name)
            model_used[(v.st_dev,v.st_ino)]=max(v.st_size,v.st_blocks*512)
        current={};counts={}
        for role,root in [('physical',self.physical),('neural',self.root),('owner',self.owner)]:
            rows,allocated=inventory(root,uid=self.uid)
            for path,item in rows.items():
                if item['type']=='symlink':
                    allowed=role=='neural'and(path==str(self.root/'model_views/D25')and item['canonical_target']==str(model)
                        or path==str(self.root/'model_inputs/nvidia/Cosmos-Reason2-2B')and item['canonical_target'].endswith(
                            '/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'))
                    need(allowed,'ONLY_ORIGINAL_PINNED_READONLY_INPUT_ALIASES')
                need(not path.endswith('.safetensors'),'NO_NEW_TENSOR_COPY_IN_CURRENT_OUTPUT')
            current[role]=allocated;counts[role]=len(rows)
        def total(*roles):
            unique=dict(model_used)
            for role in roles:
                for key,size in current[role].items():unique[key]=max(unique.get(key,0),size)
            return sum(unique.values())
        # Neural/controller and physical outputs retain the two original gross
        # financial buckets, even though both now physically reside on Core.
        site=total('neural','owner');core=total('physical')
        need(site<=self.marker['paid_IO_credit']['Hopper_storage_bytes']
             and core<=self.marker['paid_IO_credit']['Core_storage_bytes'],'ORIGINAL_PAID_GROSS_OUTPUT_ENVELOPES')
        self.proof=dict(schema='D25_CURRENT_ACTUAL_PAID_OUTPUT_GUARD_V1',request_id=self.marker['replacement_request_id'],
            model_inode_members=len(model_used),original_site_gross_current_bytes=site,original_Core_gross_current_bytes=core,
            entries=counts,new_tensor_copies=0,model_streams=0,canonical_GLOBAL_is_fresh=False,native_permission=False)
        self.last=self.clock();return self.proof
