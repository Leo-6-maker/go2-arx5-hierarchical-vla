"""Use the established qselect/individual-qstat reader on either owned site."""
from pathlib import Path
import subprocess

from scripts.pi0292.allocation_admission import owned_jobs


PREFIXES=('/opt/pbs/bin','/cm/shared/apps/pbspro/current/bin')


def owned_queue(user, *, prefix, query=subprocess.run):
    prefix=str(Path(prefix))
    if prefix not in PREFIXES:
        raise PermissionError('REGISTERED_PBS_PREFIX_REQUIRED')

    def on_site(argv,**kwargs):
        argv=list(argv);name=Path(argv[0]).name
        if name not in ('qselect','qstat'):
            raise PermissionError('OWN_QUEUE_READER_IS_READ_ONLY')
        argv[0]=str(Path(prefix)/name)
        return query(argv,**kwargs)

    return owned_jobs(user,query=on_site)
