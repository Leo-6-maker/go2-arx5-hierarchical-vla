"""Compare declared frozen tensor payloads as bytes; no model or tensor library."""
import argparse,hashlib,json,math,struct,tempfile
from pathlib import Path

def header(path):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        assert 0<size<=16*2**20
        value=json.loads(f.read(size))
    return 8+size,value

def digest(path,start,end):
    assert 0<=start<=end<=path.stat().st_size
    h=hashlib.sha256()
    with path.open('rb') as f:
        f.seek(start);remaining=end-start
        while remaining:
            chunk=f.read(min(8<<20,remaining));assert chunk
            h.update(chunk);remaining-=len(chunk)
    return h.hexdigest()

def compare(parent,model,scope):
    indexes=[json.loads((p/'model.safetensors.index.json').read_text())['weight_map'] for p in (parent,model)]
    frozen=scope['frozen'];assert len(frozen)==559 and len({r['name'] for r in frozen})==559
    metadata={};rows=[]
    for r in frozen:
        values=[]
        for root,index in zip((parent,model),indexes):
            assert Path(index[r['name']]).name==index[r['name']]
            p=root/index[r['name']]
            if p not in metadata:metadata[p]=header(p)
            base,h=metadata[p];entry=h[r['name']]
            assert entry['shape']==r['shape'] and entry['dtype']=={'torch.bfloat16':'BF16','torch.float32':'F32'}[r['dtype']]
            lo,hi=entry['data_offsets'];assert hi-lo==math.prod(r['shape'])*{'BF16':2,'F32':4}[entry['dtype']]
            values.append((hi-lo,digest(p,base+lo,base+hi)))
        assert values[0]==values[1],r['name']
        rows.append(dict(name=r['name'],bytes=values[0][0],sha256=values[0][1]))
    return dict(status='PASS_559_FROZEN_TENSOR_PAYLOAD_BYTES_IDENTICAL',parent=str(parent),model=str(model),rows=rows,new_model_queries=0,new_gradient_queries=0,new_optimizer_steps=0)

def check():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        p=Path(tmp)/'toy.safetensors';h=json.dumps({'x':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode();body=b'\0\0\x80?'
        p.write_bytes(struct.pack('<Q',len(h))+h+body);base,entries=header(p)
        assert entries['x']['shape']==[1] and digest(p,base,base+4)==hashlib.sha256(body).hexdigest()
        try:digest(p,base,base+5)
        except AssertionError:pass
        else:raise AssertionError('Out-of-file tensor range accepted')
        scope={'frozen':[dict(name=str(i),shape=[1],dtype='torch.float32') for i in range(559)]}
        h=json.dumps({str(i):dict(dtype='F32',shape=[1],data_offsets=[4*i,4*i+4]) for i in range(559)}).encode()
        payload=struct.pack('<Q',len(h))+h+body*559
        left,right=Path(tmp)/'left',Path(tmp)/'right'
        for root in (left,right):
            root.mkdir();(root/'shard').write_bytes(payload);(root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{str(i):'shard' for i in range(559)}}))
        assert len(compare(left,right,scope)['rows'])==559
        (right/'shard').write_bytes(payload[:-1]+b'!')
        try:compare(left,right,scope)
        except AssertionError:pass
        else:raise AssertionError('Changed frozen bytes accepted')
    print('PASS_STDLIB_SAFETENSORS_HEADER_PAYLOAD_BOUNDARY_CHECK')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path);p.add_argument('--model',type=Path);p.add_argument('--scope',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:check()
    else:
        assert a.parent and a.model and a.scope and a.output
        result=compare(a.parent,a.model,json.loads(a.scope.read_text()));result['scope_sha256']=hashlib.sha256(a.scope.read_bytes()).hexdigest()
        tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(result,sort_keys=True)+'\n');tmp.replace(a.output);print(result['status'])
