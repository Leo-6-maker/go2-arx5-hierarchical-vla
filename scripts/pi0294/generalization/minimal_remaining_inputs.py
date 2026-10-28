"""Read the actual closed prefix as inputs, without restreaming its tensors.

The registered new manifest must seal the private custody plan and full RESULT
and closure bytes. No native admission, writer, model or checkpoint loader is
implemented here. Changed files are refused; they are not silently reaudited.
"""
from copy import deepcopy
import json,os,stat
from pathlib import Path
from . import minimal_remaining_contract as r,minimal_model as model
from . import minimal_saved_prefix_audit as old

PLAN='plans/MINIMAL_SAVED_PREFIX_CUSTODY.private.json'
SCHEMA='ARX5_MINIMAL_REAL_SAVED_PREFIX_READONLY_CUSTODY_V1'
AUDIT_MANIFEST='ff3567548e483bb4a73bfbd1c55af428bb9d9e9df634d2c616b7162dad92245d'
need=r.e.need


def own_body(path,*,immutable=True):
    path=Path(path);s=path.lstat()
    need(path.is_absolute()and path.resolve()==path and stat.S_ISREG(s.st_mode)
        and s.st_uid==os.getuid()and(not immutable or stat.S_IMODE(s.st_mode)==0o400),
        'PREFIX_INPUT_OWN_REGULAR_EXACT_FILE_NO_ALIAS')
    need(s.st_size<=2*1024**2,'PREFIX_INPUT_SMALL_METADATA_ONLY_NOT_TENSOR')
    body=path.read_bytes()
    after=path.lstat()
    need(all(getattr(after,k)==getattr(s,k)for k in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_uid')),
        'PREFIX_INPUT_UNCHANGED_DURING_METADATA_READ')
    return body


class Inputs:
    def __init__(self,root,manifest,*,guard):
        self.root=Path(root);self.guard=guard;need(callable(guard),'PREFIX_INPUT_EXTERNAL_NATIVE_GUARD')
        sealed=manifest['execution']['sealed_inputs']
        self.guard();pb=own_body(self.root/PLAN)
        need(r.e.sha(pb)==sealed.get(PLAN),'PREFIX_INPUT_NEW_MANIFEST_SEALS_CUSTODY')
        p=json.loads(pb);self.plan=p
        need(p['schema']==SCHEMA and p['result_sha256']==r.RESULT and p['closure_sha256']==r.CLOSED
            and p['audit_manifest_sha256']==AUDIT_MANIFEST and p['science_manifest_sha256']==old.MANIFEST
            and p['science_source_sha']==r.OLD_SOURCE and p['new_model_queries_updates_physics']==0,
            'PREFIX_INPUT_EXACT_CLOSED_NATIVE_AUDIT_IDENTITY')
        bodies=[]
        for name,pin in((r.PREFIX,r.RESULT),(r.CLOSURE,r.CLOSED)):
            body=own_body(self.root/name)
            need(r.e.sha(body)==sealed.get(name)==pin,'PREFIX_INPUT_FULL_RESULT_AND_CLOSURE_SEALED')
            bodies.append(body)
        self.binding=r.saved_prefix(*bodies);self.result=json.loads(bodies[0])
        self.science=Path(p['science_root']);self.audit=Path(p['audit_root'])
        sb=own_body(self.science/'RUN_MANIFEST.json');ab=own_body(self.audit/'RUN_MANIFEST.json')
        need(r.e.sha(sb)==old.MANIFEST and r.e.sha(ab)==AUDIT_MANIFEST
            and json.loads(sb)['code']['vla_commit']==r.OLD_SOURCE
            and json.loads(ab)['plan']['science_root']==str(self.science),
            'PREFIX_INPUT_ORIGINAL_CLOSED_SCIENCE_AND_AUDITOR_ROOTS')
        need(own_body(self.audit/'RESULT.json')==bodies[0],'PREFIX_INPUT_ACTUAL_ORIGINAL_AUDIT_RESULT')
        for folder,pin in((self.science,r.authority.LEASES['science']),(self.audit,r.AUDIT_LEASE)):
            binding=json.loads(own_body(folder/'OUTPUT_BINDING.private.json'))
            lb=own_body(Path(binding['lease']),immutable=False)
            need(r.e.sha(lb)==pin and json.loads(lb)['status']=='CLOSED','PREFIX_INPUT_ACTUAL_CLOSED_LEASE_BYTES')
        wb=own_body(self.audit/'FULL_STREAMED_PREFIX_WITNESSES.json')
        rb=own_body(self.audit/'POST_F_LOGIN_RAW_PARENT_FULL_STREAM_AUDIT.json')
        need(r.e.sha(wb)==self.result['full_streamed_witnesses_sha256']
            and r.e.sha(rb)==self.result['raw_parent_post_F_audit_sha256'],
            'PREFIX_INPUT_ORIGINAL_FULL_STREAM_WITNESSES_NOT_HEADER_ASSERTION')
        witnesses=json.loads(wb)
        need(witnesses['all_paths_rechecked']is witnesses['all_unique_inodes_independently_rehashed']is True,
            'PREFIX_INPUT_REAL_SECOND_STREAM_COMPLETE')
        self.paths={row['path']:row for row in witnesses['paths']}
        need(len(self.paths)==len(witnesses['paths'])and self.paths,'PREFIX_INPUT_UNIQUE_ORIGINAL_WITNESSES')
        self.raw=json.loads(rb)
        self.raw_checkpoint=Path(json.loads(own_body(self.science/'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json'))['checkpoint'])
        self.verify_current()

    def verify_current(self):
        self.guard()
        for i,(name,row)in enumerate(self.paths.items()):
            if i%32==0:self.guard()
            p=Path(name);s=p.lstat()
            need(p.is_relative_to(self.science)and p.resolve()==p and stat.S_ISREG(s.st_mode)
                and s.st_uid==os.getuid()and not s.st_mode&0o222
                and(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(row[k]for k in('device','inode','bytes','mtime_ns','ctime_ns')),
                'PREFIX_INPUT_CHANGED_SAVED_PREFIX_REQUIRES_NEW_LEGAL_AUDIT')
        model.unchanged_raw_parent(self.raw_checkpoint,self.raw)
        return deepcopy(self.raw)

    def E2_zeros(self,*,input_fingerprint):
        from . import minimal_native_audit as audit
        self.verify_current();sid=r.sessions()[0]['id'];rows=[]
        def small_stream(path):
            self.guard();name=str(path);row=self.paths.get(name)
            need(row is not None and row['bytes']<=1024**2,'PREFIX_INPUT_ONLY_ORIGINAL_SMALL_SAVED_DRAW')
            body=own_body(path);digest=r.e.sha(body)
            need(digest==row['sha256'],'PREFIX_INPUT_SAVED_DRAW_BYTES_CHANGED')
            return digest
        for i in range(15):
            path=self.science/'evidence/MINIMAL_FIVE_SESSIONS'/sid/'draws'/('zero-'+str(i).zfill(2)+'.npz')
            digest=audit.draw(path,small_stream,seed=29400+i,input_fingerprint=input_fingerprint)
            rows.append(dict(path=str(path),sha256=digest,noise_seed=29400+i))
        self.verify_current()
        return dict(schema='ARX5_MINIMAL_FIFTEEN_SAVED_E2_ZEROS_CURRENT_READONLY_INPUT_V1',
            prefix_result_sha256=r.RESULT,session_id=sid,original_draws=rows,
            actual_saved_queries_reused=15,new_model_queries=0,new_effect_query_links=0,
            source_kind='ALREADY_EXECUTED_AND_INDEPENDENTLY_AUDITED_OUTPUTS')
