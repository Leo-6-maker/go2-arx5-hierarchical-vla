"""Public checkpoint export: opaque weights, parsed metadata, no pickle execution."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickletools
import re
import struct
import zipfile

COSMOS='nvidia/Cosmos-Reason2-2B'
PRIVATE=re.compile(r'/(?:scratch|home|var/cache)/|hf_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9]|\b10\.(?:248|249)\.')


def reject_private(value):
    if isinstance(value,dict):
        for key,item in value.items():reject_private(key);reject_private(item)
    elif isinstance(value,(list,tuple)):
        for item in value:reject_private(item)
    elif isinstance(value,str) and PRIVATE.search(value):
        raise ValueError('PUBLIC_METADATA_PRIVATE_STRING_REFUSED')


def public_config(value,*,processor=False):
    value=json.loads(json.dumps(value))
    target=value['processor_kwargs'] if processor else value
    if not isinstance(target.get('model_name'),str) or not PRIVATE.search(target['model_name']):
        raise ValueError('EXPECTED_PRIVATE_COSMOS_BINDING_REQUIRED')
    target['model_name']=COSMOS
    reject_private(value)
    return value


def relocate_pickle(data,root):
    """Rewrite protocol-2 Unicode operands only; all executable opcodes retained."""
    ops=list(pickletools.genops(data))
    if not ops or ops[0][0].name!='PROTO' or ops[0][1]!=2 or any(o.name=='FRAME' for o,a,p in ops):
        raise ValueError('ONLY_UNFRAMED_PROTOCOL2_ARGS_SUPPORTED')
    pieces=[];changed=0;expected=[]
    for i,(op,arg,pos) in enumerate(ops):
        end=ops[i+1][2] if i+1<len(ops) else len(data)
        new=arg
        if isinstance(arg,str) and PRIVATE.search(arg):
            if op.name!='BINUNICODE' or not arg.startswith(str(root)+'/'):
                raise ValueError('UNEXPECTED_PRIVATE_PICKLE_STRING')
            relative=PurePosixPath(arg).relative_to(PurePosixPath(root))
            if '..' in relative.parts:raise ValueError('PUBLIC_ARGS_PATH_ESCAPE')
            new='PUBLIC_RUN_ROOT/'+relative.as_posix();encoded=new.encode('utf-8')
            pieces.append(b'X'+struct.pack('<I',len(encoded))+encoded);changed+=1
        else:pieces.append(data[pos:end])
        expected.append((op.name,new))
    if changed!=2:raise ValueError('EXACT_TWO_TRAINER_PATH_OPERANDS_REQUIRED')
    result=b''.join(pieces)
    if [(o.name,a) for o,a,p in pickletools.genops(result)]!=expected:
        raise ValueError('PUBLIC_ARGS_OPCODE_OR_PARAMETER_CHANGE')
    for op,arg,pos in pickletools.genops(result):
        if isinstance(arg,str):reject_private(arg)
    return result,changed


def relocate_args(source,target,root):
    with zipfile.ZipFile(source) as original:
        names=original.namelist()
        pickles=[n for n in names if n.endswith('/data.pkl')]
        if len(pickles)!=1 or any('/data/' in n for n in names):
            raise ValueError('ARGS_ARCHIVE_MUST_NOT_CONTAIN_TENSOR_STORAGE')
        data,count=relocate_pickle(original.read(pickles[0]),root)
        with zipfile.ZipFile(target,'x') as public:
            for entry in original.infolist():
                public.writestr(entry,data if entry.filename==pickles[0] else original.read(entry.filename))
    return count


def audit_tensor_pickle(path):
    """Inspect string operands without loading optimizer/RNG objects or tensors."""
    with zipfile.ZipFile(path) as archive:
        p=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(p)!=1:raise ValueError('ONE_TORCH_PICKLE_HEADER_REQUIRED')
        for op,arg,pos in pickletools.genops(archive.read(p[0])):
            if isinstance(arg,str):reject_private(arg)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
