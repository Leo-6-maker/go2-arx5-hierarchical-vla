"""Read-only stat inventory confined to PI0286 namespaces, no array or blob reads."""
import json,sys,time,concurrent.futures
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution')
from coordinator import remote
OUT=Path('reports/PI0286/final')
SCRIPT=r'''
import os,json,time,stat,collections
from pathlib import Path
root=Path(ROOT); roots=[root]
if HOST=='core':roots.append(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0286'))
allowed=[str(p) for p in roots];seen_dirs=set();inodes={};aliases=[];errors=[];groups={};directories={}
def permitted(p):return any(str(p)==x or str(p).startswith(x+'/') for x in allowed)
def group(p):
 for base in roots:
  try:parts=p.relative_to(base).parts;break
  except ValueError:continue
 else:return 'UNKNOWN'
 tier='SSD/' if str(base).startswith('/var/cache') else 'ROOT/'
 n=3 if parts and parts[0]=='learning' else 2 if parts and parts[0] in ('models','delivery','durable','archive','source') else 1
 return tier+'/'.join(parts[:n])
def add(p,s):
 g=group(p);d=groups.setdefault(g,dict(files=0,logical_bytes=0,path_allocated_bytes=0,unique_allocated_bytes=0,unique_logical_bytes=0,external_original_inode_bytes=0))
 d['files']+=1;d['logical_bytes']+=s.st_size;d['path_allocated_bytes']+=s.st_blocks*512
 key=(s.st_dev,s.st_ino)
 if key not in inodes:
  original=key in original_inodes
  inodes[key]=dict(path=str(p),size=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,original_dependency=original,aliases=[])
  if original:d['external_original_inode_bytes']+=s.st_size
  else:d['unique_allocated_bytes']+=s.st_blocks*512;d['unique_logical_bytes']+=s.st_size
 else:inodes[key]['aliases'].append(str(p))
def walk(p):
 try:s=p.lstat()
 except OSError as e:errors.append([str(p),str(e)]);return
 if stat.S_ISLNK(s.st_mode):
  target=p.resolve();v=dict(path=str(p),target=str(target),owned_target=permitted(target),link_bytes=s.st_size)
  try:
   t=p.stat();v.update(target_bytes=t.st_size,target_is_directory=stat.S_ISDIR(t.st_mode),dev=t.st_dev,ino=t.st_ino)
  except OSError as e:v['error']=str(e)
  aliases.append(v)
  # Explicit roots traversed separately; don't follow alias paths and double-count.
  return
 if stat.S_ISDIR(s.st_mode):
  key=(s.st_dev,s.st_ino)
  if key in seen_dirs:return
  seen_dirs.add(key);directories[group(p)]=directories.get(group(p),0)+s.st_blocks*512
  try:
   for child in p.iterdir():walk(child)
  except OSError as e:errors.append([str(p),str(e)])
 elif stat.S_ISREG(s.st_mode):add(p,s)
original_inodes=set();parent= root/'design/PARENT_READBACK.json';parent_identity=None
if parent.exists():
 v=json.loads(parent.read_text());p=Path(v['parent']);members=[]
 for n,x in v['files'].items():
  try:s=(p/n).stat();original_inodes.add((s.st_dev,s.st_ino));members.append(dict(path=str(p/n),bytes=s.st_size,allocated_bytes=s.st_blocks*512))
  except OSError as e:errors.append([str(p/n),str(e)])
 parent_identity=dict(path=str(p),original_existing_members=members,not_traversed=True)
start=time.time()
for p in roots:
 if p.exists():walk(p)
own=[v for v in inodes.values() if not v['original_dependency']]
large=[dict(v,dev=k[0],ino=k[1]) for k,v in inodes.items() if v['size']>=100<<20]
result=dict(status='READ_ONLY_PI0286_SCOPED_STAT_SNAPSHOT_NOT_ATOMIC_LIVE_OUTPUTS',host=HOST,epoch=time.time(),elapsed_seconds=time.time()-start,roots=allowed,groups=groups,regular_file_paths=sum(d['files'] for d in groups.values()),regular_file_logical_bytes=sum(d['logical_bytes'] for d in groups.values()),owned_unique_file_logical_bytes=sum(v['size'] for v in own),owned_unique_file_allocated_bytes=sum(v['allocated_bytes'] for v in own),directory_allocated_bytes=sum(directories.values()),original_dependency_inode_bytes=sum(v['size'] for v in inodes.values() if v['original_dependency']),hardlink_or_alias_duplicate_paths=sum(len(v['aliases']) for v in inodes.values()),symlink_aliases=aliases,original_parent=parent_identity,large_files=large,errors=errors,model=0,physics=0,optimizer=0,deleted=0,content_hash_reads=0)
print(json.dumps(result))
'''
def scan(host):
 root='/home/liuyu-nfs/Projects/go2-arx5/PI0286' if host=='core' else '/scratch/e1192548/Projects/go2-arx5/experiments/PI0286'
 v=json.loads(remote(host,'ROOT='+repr(root)+'\nHOST='+repr(host)+'\n'+SCRIPT));(OUT/('PI0286_OWNED_STORAGE_'+host.upper()+'.json')).write_text(json.dumps(v,indent=2)+'\n');return v
if __name__=='__main__':
 values=list(concurrent.futures.ThreadPoolExecutor(max_workers=2).map(scan,('core','hop')))
 total=sum(v['owned_unique_file_allocated_bytes']+v['directory_allocated_bytes'] for v in values)
 print(json.dumps(dict(total_allocated_bytes=total,total_GiB=total/(1<<30),cap_bytes=100<<30,headroom_bytes=(100<<30)-total,hosts=[dict(host=v['host'],allocated=v['owned_unique_file_allocated_bytes'],logical=v['owned_unique_file_logical_bytes'],large_files=len(v['large_files']),errors=v['errors']) for v in values])))
