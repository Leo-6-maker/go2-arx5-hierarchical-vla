"""Bounded memo of pure validation of exact complete canonical ledger bytes.

No process, clock freshness, queue, quota, filesystem or native release result
is cached. Any ledger byte change forces validation again. Exceptions are not
cached; callers receive independent copies and cannot mutate a cached proof.
"""
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import hashlib,json

MAX_ENTRIES=8
MAX_RESULT_BYTES=32*1024**2


def exact_ledger_validation(fn):
    cache=OrderedDict();sizes={}
    @wraps(fn)
    def validate(g,*,generation=None):
        # This digest is private to this memo, never a protocol/artifact SHA.
        # Compact stdlib JSON preserves order, numeric types and every nested
        # value, while using the C encoder instead of the slow indented Python
        # iterator on the actual 15MB retained campaign history.
        body=json.dumps(g,separators=(',',':'),allow_nan=False).encode()
        key=(hashlib.sha256(body).digest(),len(body),generation)
        if key in cache:
            cache.move_to_end(key);return deepcopy(cache[key])
        value=fn(g,generation=generation)
        encoded=json.dumps(value,allow_nan=False).encode()
        if len(encoded)<=MAX_RESULT_BYTES:
            while cache and(len(cache)>=MAX_ENTRIES or sum(sizes.values())+len(encoded)>MAX_RESULT_BYTES):
                old,_=cache.popitem(last=False);sizes.pop(old)
            cache[key]=deepcopy(value);sizes[key]=len(encoded)
        return value
    return validate
