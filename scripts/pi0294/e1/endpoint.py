"""E1-bound registry; original protected file endpoint operations unchanged."""
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_endpoint import Endpoint as OriginalEndpoint,inside
from .scope import validate


class Endpoint(OriginalEndpoint):
    def __init__(self,root,registry_relative,registry_sha256,role,*,now=time.time):
        self.root,self.role,self.now=Path(root),role,now
        if role not in ['core','hopper']:raise PermissionError('E1_EXACT_ENDPOINT_ROLE')
        m=validate(self.root,'4080-simulation' if role=='core' else 'hopper-neural')
        path=inside(self.root,registry_relative)
        if sha(path)!=registry_sha256:raise PermissionError('E1_EXACT_SEALED_REGISTRY')
        self.registry=read(path)
        if (self.registry['run_uuid']!=self.root.name or self.registry[role+'_source_sha']!=m['code']['vla_commit']
                or self.registry[role+'_manifest_sha256']!=sha(self.root/'RUN_MANIFEST.json')
                or now()>=self.registry['deadline_epoch'] or any(c['card']['split']!='TRAIN' for c in self.registry['cases'].values())):
            raise PermissionError('E1_RUN_SOURCE_TRAIN_SCOPE_OR_DEADLINE')
