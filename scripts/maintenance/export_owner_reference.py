"""Publish path-redacted, non-executable owner source references, never data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


SECRET_PATTERNS=(
    r'\bhf_[A-Za-z0-9]{20,}\b',r'\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{20,}\b',r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    r'\bAKIA[A-Z0-9]{16}\b',
)
GUARD='raise SystemExit("OWNER_REFERENCE_ONLY_NOT_A_RUN_ENTRY_OR_AUTHORIZATION")\n'


def check_secrets(text):
    if any(re.search(pattern,text) for pattern in SECRET_PATTERNS):
        raise ValueError('POSSIBLE_SECRET_LITERAL_REFUSE_PUBLICATION')
    if re.search(r'(?im)^\s*(?:password|passwd|api_key|access_token|hf_token)\s*=\s*[\"\'][^\"\']{8,}[\"\']',text):
        raise ValueError('CREDENTIAL_LITERAL_REFUSE_PUBLICATION')
    try:tree=ast.parse(text)
    except SyntaxError:return
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            if len(node.value)>512 and re.fullmatch(r'[A-Za-z0-9+/=]+',node.value):
                raise ValueError('OPAQUE_STATIC_BASE64_REFUSE_PUBLICATION')


def redact(text, replacements):
    check_secrets(text)
    if not replacements or any(not old or not new.startswith('@') for old,new in replacements.items()):
        raise ValueError('PRIVATE_REDACTION_MAP_WITH_ROLE_PLACEHOLDERS_REQUIRED')
    counts={}
    for old,new in sorted(replacements.items(),key=lambda row:-len(row[0])):
        count=text.count(old)
        if count:counts[new]=counts.get(new,0)+count;text=text.replace(old,new)
    if any(old in text for old in replacements):raise ValueError('PRIVATE_FRAGMENT_REMAINED')
    if re.search(r'/(?:home|scratch|var/cache/users)/[A-Za-z0-9_.-]+',text):
        raise ValueError('UNREVIEWED_PRIVATE_ABSOLUTE_PATH')
    if re.search(r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b',text):
        raise ValueError('UNREVIEWED_PRIVATE_NETWORK_ADDRESS')
    check_secrets(text)
    return text,counts


def guarded(text):
    text=''.join(line.rstrip()+'\n' for line in text.splitlines())
    try:
        tree=ast.parse(text)
        imports=[n.end_lineno for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__']
        after=max(imports,default=0);syntax='PARSE_PASS'
    except SyntaxError:
        after=0;syntax='HISTORICAL_SOURCE_SYNTAX_ERROR_RETAINED'
    lines=text.splitlines(keepends=True)
    body=''.join(lines[:after])+GUARD+''.join(lines[after:])
    return '# SANITIZED OWNER REFERENCE; EXACT ORIGINAL STAYS PRIVATE.\n# Do not execute, deploy, or treat historical constants as a current GO.\n'+body,syntax


def export(root, output, replacements):
    root=Path(root).resolve(strict=True);output=Path(output)
    if output.exists():raise FileExistsError('PRESERVE_PREVIOUS_REFERENCE_SNAPSHOT')
    pending=[]
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' in path.parts:continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('SOURCE_LINK_OR_PATH_ESCAPE')
        original=path.read_bytes();text=original.decode('utf-8')
        clean,counts=redact(text,replacements);body,syntax=guarded(clean)
        if syntax=='PARSE_PASS':compile(body,str(path.relative_to(root)),'exec')
        pending.append((path.relative_to(root).as_posix()+'.txt',original,body.encode(),counts,syntax))
    if not pending:raise ValueError('NO_OWNER_PYTHON_SOURCES_FOUND')
    output.mkdir(parents=True)
    entries=[]
    for relative,original,data,counts,syntax in pending:
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        entries.append(dict(path=relative,original_source_sha256=hashlib.sha256(original).hexdigest(),
            sanitized_source_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
            role_redaction_counts=counts,syntax=syntax,executable=False))
    value=dict(schema='SANITIZED_OWNER_SOURCE_REFERENCE_V1',status='SOURCE_REFERENCE_NOT_RUNNABLE_DEPLOYMENT',
        source_file_count=len(entries),total_bytes=sum(row['bytes'] for row in entries),files=entries,
        exact_private_originals_retained=True,private_json_logs_data_weights_credentials_excluded=True,
        runtime_source_changed=False,scientific_GO_created=False)
    with (output/'MANIFEST.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--private-source-root',required=True);p.add_argument('--private-redaction-map',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    value=export(a.private_source_root,a.output,json.loads(Path(a.private_redaction_map).read_text()))
    print(json.dumps({key:value[key] for key in ('status','source_file_count','total_bytes')}))
