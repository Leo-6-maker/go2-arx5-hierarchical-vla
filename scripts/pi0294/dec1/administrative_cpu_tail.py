"""Correct one known legacy financial tail's absolute CPU-limit bug.

Pure source transformation only; no IO, permission, connection or retry.
The existing process CPU is included in the absolute POSIX limit. Flooring
keeps additional soft/hard headroom at most the original 8/10 seconds.
"""

OLD='resource.setrlimit(resource.RLIMIT_CPU,(8,10))'
NEW=('assert 0<=began and began+10<=1800;'
     'resource.setrlimit(resource.RLIMIT_CPU,(int(began)+8,int(began)+10))')


def bind(program):
    if (type(program) is not str or program.count(OLD)!=1
            or program.count('began=time.process_time();')!=1
            or program.index('began=time.process_time();')>program.index(OLD)):
        raise PermissionError('EXACT_LEGACY_FINANCIAL_TAIL_ONLY_NO_GENERIC_LIMIT_OVERRIDE')
    return program.replace(OLD,NEW,1)
