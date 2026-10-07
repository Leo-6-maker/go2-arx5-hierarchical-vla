"""Lossless replacement bytes over the local donor; never tensor subtraction/addition."""
import argparse,hashlib,json,os,shutil,struct,tempfile,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert 0<n<=16<<20
        text=f.read(n);assert len(text)==n
    return 8+n,json.loads(text)

def digest(path,offset,size):
    h=hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while size:
            b=f.read(min(size,8<<20));assert b;h.update(b);size-=len(b)
    return h.hexdigest()

def pack(parent,model,dest):
    started=time.time();ready=json.loads((model/'READY.json').read_text())
    if (dest/'READY.json').exists():return json.loads((dest/'OVERLAY.json').read_text())
    temp=dest.with_name(dest.name+'.partial');temp.mkdir(parents=True,exist_ok=False)
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    assert index==json.loads((parent/'model.safetensors.index.json').read_text())['weight_map']
    rows=[];files=[]
    for name in sorted(set(index.values())):
        assert Path(name).name==name
        a,b=parent/name,model/name;base,h=header(a);other,hh=header(b)
        # ponytail: identical serialization layout only; use direct transfer if it differs.
        assert base==other and h==hh and a.stat().st_size==b.stat().st_size
        previous=0;replacement=temp/(name+'.replacement')
        with replacement.open('wb') as out:
            for key,entry in sorted(((k,v) for k,v in h.items() if k!='__metadata__'),key=lambda x:x[1]['data_offsets']):
                assert index[key]==name
                lo,hi=entry['data_offsets'];assert lo==previous and hi>=lo;previous=hi
                ah,bh=digest(a,base+lo,hi-lo),digest(b,base+lo,hi-lo)
                row=dict(name=key,shard=name,dtype=entry['dtype'],shape=entry['shape'],file_offset=base+lo,bytes=hi-lo,parent_sha256=ah,source_sha256=bh,changed=ah!=bh)
                if ah!=bh:
                    row['replacement_offset']=out.tell()
                    with b.open('rb') as inp:
                        inp.seek(base+lo);remaining=hi-lo
                        while remaining:
                            block=inp.read(min(remaining,8<<20));assert block;out.write(block);remaining-=len(block)
                rows.append(row)
            assert base+previous==b.stat().st_size
        files.append(dict(path=replacement.name,bytes=replacement.stat().st_size,sha256=sha(replacement)))
    assert set(index)=={r['name'] for r in rows}
    for row in ready['files']:
        if row['path'].endswith('.safetensors'):continue
        shutil.copyfile(model/row['path'],temp/row['path']);files.append(row)
    value=dict(status='EXACT_REPLACEMENT_TENSOR_TRANSPORT',point=ready['point'],source_ready=ready,tensors=rows,changed_tensors=sum(r['changed'] for r in rows),replacement_bytes=sum(r['bytes'] for r in rows if r['changed']),pack_wall_s=time.time()-started)
    (temp/'OVERLAY.json').write_text(json.dumps(value,sort_keys=True)+chr(10));files.append(dict(path='OVERLAY.json',bytes=(temp/'OVERLAY.json').stat().st_size,sha256=sha(temp/'OVERLAY.json')))
    (temp/'READY.json').write_text(json.dumps(dict(status='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT',point=ready['point'],files=files),sort_keys=True)+chr(10));temp.replace(dest)
    return value

def restore(parent,overlay,dest):
    started=time.time();value=json.loads((overlay/'OVERLAY.json').read_text());ready=value['source_ready']
    temp=dest.with_name(dest.name+'.partial');temp.mkdir(parents=True,exist_ok=False)
    for row in ready['files']:
        name=row['path'];assert Path(name).name==name
        if not name.endswith('.safetensors'):shutil.copyfile(overlay/name,temp/name);continue
        shutil.copyfile(parent/name,temp/name)
        with (temp/name).open('r+b') as out,(overlay/(name+'.replacement')).open('rb') as inp:
            for tensor in value['tensors']:
                if tensor['shard']!=name or not tensor['changed']:continue
                assert digest(parent/name,tensor['file_offset'],tensor['bytes'])==tensor['parent_sha256']
                inp.seek(tensor['replacement_offset']);out.seek(tensor['file_offset']);remaining=tensor['bytes']
                while remaining:
                    block=inp.read(min(remaining,8<<20));assert block;out.write(block);remaining-=len(block)
            out.flush();os.fsync(out.fileno())
    for row in ready['files']:assert (temp/row['path']).stat().st_size==row['bytes'] and sha(temp/row['path'])==row['sha256'],row['path']
    (temp/'READY.json').write_text(json.dumps(ready,sort_keys=True)+chr(10));temp.replace(dest)
    return dict(status='PASS_FULL_SOURCE_FILES_AFTER_REPLACEMENT_RECONSTRUCTION',point=ready['point'],tensors=len(value['tensors']),changed_tensors=value['changed_tensors'],replacement_bytes=value['replacement_bytes'],restore_wall_s=time.time()-started,new_model_queries=0)

def check():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
        root=Path(folder);a,b,c,d=[root/n for n in ('parent','model','overlay','restored')];a.mkdir();b.mkdir()
        h=json.dumps({'x':dict(dtype='F32',shape=[1],data_offsets=[0,4]),'buffer':dict(dtype='I32',shape=[1],data_offsets=[4,8])}).encode()
        prefix=struct.pack('<Q',len(h))+h
        for r,payload in [(a,b'12345678'),(b,b'1234abcd')]:
            (r/'model.safetensors').write_bytes(prefix+payload);(r/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'x':'model.safetensors','buffer':'model.safetensors'})))
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in b.iterdir()]
        (b/'READY.json').write_text(json.dumps(dict(point='toy',files=files)))
        v=pack(a,b,c);assert v['changed_tensors']==1 and v['replacement_bytes']==4
        restore(a,c,d);assert (d/'model.safetensors').read_bytes()==(b/'model.safetensors').read_bytes()
        (c/'model.safetensors.replacement').write_bytes(b'bad!')
        try:restore(a,c,root/'wrong')
        except AssertionError:pass
        else:raise AssertionError('Corrupt replacement accepted')
    print('PASS_EXACT_BYTES_CHANGED_BUFFER_AND_CORRUPTION_REJECTION')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['pack','restore','check']);p.add_argument('--parent',type=Path);p.add_argument('--source',type=Path);p.add_argument('--dest',type=Path);a=p.parse_args()
    if a.operation=='check':check()
    else:
        v=globals()[a.operation](a.parent,a.source,a.dest);print(json.dumps({k:v[k] for k in v if k not in ('tensors','source_ready')}) if a.operation=='pack' else json.dumps(v),flush=True)
