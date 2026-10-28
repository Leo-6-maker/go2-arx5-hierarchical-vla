"""Historical source90 recovery2 contract, sharing the original strict checks.

The exact historical source/run stay pinned. A different registered attempt
uses its separately qualified binding; it cannot acquire this old identity.
"""
from .b_recovery2_completion_contract import (
    CompletionContract,need,recovery,b,saved,a40_allocation,digest)
SOURCE='90d357a74d3d8e1f033ca978f16c5f3191a087f9'
RUN='vap1-a40-b-recovery2-01'
_contract=CompletionContract(source=SOURCE,run=RUN)
audit=_contract.audit
effects=_contract.effects
expected=_contract.expected
native=_contract.native
origin=_contract.origin
owner=_contract.owner
query_provenance=_contract.query_provenance
settled=_contract.settled
