"""Only unchanged independently streamed old tensors may be consumed.

Sealed small receipts and native backing-filesystem identities are checked on
both login and compute. No tensor copying, chmod, re-audit or permission here.
"""
import json,os,stat
from pathlib import Path
from . import minimal_saved512_contract as c,minimal_execution as e,minimal_portable_inventory as portable
from .minimal_remaining_inputs import Inputs as PriorInputs,own_body


class Inputs:
    def __init__(self,root,manifest,*,guard):
        self.root=Path(root);self.guard=guard;guard();sealed=manifest['execution']['sealed_inputs']
        def body(name):
            b=own_body(self.root/name,maximum=8*1024**2)
            e.need(e.sha(b)==sealed.get(name),'SAVED512_CUSTODY_CURRENT_MANIFEST:'+name);return b
        rb,eb=body(c.RESULT_FILE),body(c.EXIT_FILE);self.binding=c.binding(rb,eb);self.result=json.loads(rb)
        self.plan=json.loads(body(c.PLAN));self.science=Path(self.plan['science_root']);self.audit=Path(self.plan['audit_root'])
        e.need(self.science.is_absolute()and self.science.resolve()==self.science
            and self.science.name=='arx5-minimal-raw-parent-first512-remainder-02'
            and e.sha(own_body(self.science/'RUN_MANIFEST.json'))==c.MANIFEST
            and json.loads(own_body(self.science/'RUN_MANIFEST.json'))['code']['vla_commit']==c.SOURCE,
            'SAVED512_EXACT_ORIGINAL_FAILED_SCIENCE_ROOT_AND_SOURCE')
        for name,pin,b in [('RESULT.json',c.RESULT,rb),('OS_EXIT.json',c.OS_EXIT,eb)]:
            e.need(own_body(self.audit/name,immutable=(name!='OS_EXIT.json'))==b and e.sha(b)==pin,'SAVED512_ORIGINAL_AUDIT_BYTES_NOT_RESTAMPED')
        launch=json.loads(own_body(self.audit/'LAUNCHED.json',immutable=False))
        e.need(e.sha(own_body(self.audit/'PROGRAM.py',maximum=64*1024))==c.PROGRAM
            and launch['program_sha256']==c.PROGRAM and launch['CPU_upper_s']==480
            and launch['new_GPU_PBS_models_queries_updates']==0,'SAVED512_ACTUAL_FINITE_CPU_AUDITOR_IDENTITY')
        b=json.loads(own_body(self.science/'OUTPUT_BINDING.private.json'))
        lease=own_body(Path(b['lease']),immutable=False)
        e.need(e.sha(lease)==c.scope.LEASE and json.loads(lease)['status']=='CLOSED'
            and e.sha(own_body(self.science/'control/resident-owner/TERMINAL_SETTLED.private.json'))==c.SETTLED,
            'SAVED512_ACTUAL_KNOWN_CLOSED_NOT_RESTART_OF_FAILED_OWNER')
        old_manifest=json.loads(own_body(self.science/'RUN_MANIFEST.json'))
        self.prior=PriorInputs(self.science,old_manifest,guard=guard)
        self.raw_checkpoint=self.prior.raw_checkpoint
        self.paths={r['path']:r for r in self.result['streamed_witnesses']['paths']}
        e.need(len(self.paths)==len(self.result['streamed_witnesses']['paths']) and self.paths,'SAVED512_UNIQUE_STREAMED_PATHS')
        self.verify_current()

    def verify_current(self):
        self.guard()
        e.need(portable.filesystem(self.science)==self.plan['science_filesystem'],
            'SAVED512_SAME_ACTUAL_BACKING_FILESYSTEM_NOT_DEVICE_ALIAS')
        for name,row in self.paths.items():
            p=Path(name);s=p.lstat()
            e.need(p.is_relative_to(self.science)and p.resolve()==p and stat.S_ISREG(s.st_mode)
                and s.st_uid==os.getuid()and stat.S_IMODE(s.st_mode)==0o400
                and(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(row[k]for k in('inode','bytes','mtime_ns','ctime_ns')),
                'SAVED512_EVERY_FULL_STREAM_WITNESS_UNCHANGED_OR_REJECT')
        return self.prior.verify_current()

    def E1_paths(self):
        self.verify_current();cp=self.science/'training/E1_ADAPTER_ONLY_FORMAL512/checkpoint-512'
        export=self.science/'inference/E1_ADAPTER_ONLY'
        e.need(e.sha(own_body(cp/'CHECKPOINT_COMPLETE.json'))==c.E1_CP
            and e.sha(own_body(export/'CHECKPOINT_COMPLETE.json'))==c.E1_EXPORT,
            'SAVED512_OWN512_AND_ORDINARY_FULL_MODEL_EXACT_MANIFESTS')
        return cp,export
