"""PV1 read-side use of the mature explicit NFS directory assistance."""
import os
from pathlib import Path

from scripts.pi0292.relay_protocol import uuid_string
from scripts.pi0294.relay_endpoint import Endpoint as OriginalEndpoint,inside
from scripts.pi0294.relay_io import read_available
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.mp1.directory_visibility_audit import force_server_attributes

ALGORITHM='QUALIFIED_STATX_FORCE_SYNC_AND_DIRECTORY_ENUMERATION_V2'
MAX_ENTRIES=1920


def refresh(directory):
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise PermissionError('PV1_EXISTING_REAL_PACKET_DIRECTORY_ONLY')
    force_server_attributes(directory)
    with os.scandir(directory) as entries:
        for index,_ in enumerate(entries,1):
            if index>MAX_ENTRIES:raise PermissionError('PV1_FIXED_24_CONDITION_PACKET_DIRECTORY_BOUND')


def read_visible(path):
    path=Path(path);refresh(path.parent)
    return read_available(path)


class Endpoint(OriginalEndpoint):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if read(self.root/'PV1_STAGE_BINDING.json').get('NFS_lookup_cache_assistance')!=ALGORITHM:
            raise PermissionError('PV1_PRESEALED_DIRECTORY_ASSISTANCE_REQUIRED')

    def hopper(self,operation,episode,*,query=None,teacher=None,query_uuid=None):
        if operation=='response' and self.role=='hopper':
            case=self.case(episode);uuid_string(query_uuid)
            directory=inside(self.root,case['hopper_session_relative'])
            return read_visible(inside(directory,'outbox/'+query_uuid+'.json'))
        return super().hopper(operation,episode,query=query,teacher=teacher,query_uuid=query_uuid)
