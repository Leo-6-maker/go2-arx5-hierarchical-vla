"""Fractional actual CPU debit must not erase or invent future reservation."""
import json,base64
from pathlib import Path
from decimal import Decimal
import pytest
from scripts.pi0294.pv1 import mix_core_settlement as n,mix_accounting as a,mix_core_accounting as c


def test_exact_fractional_envelope_release_has_no_binary_ulp_deficit():
 assert 529610.205680545-(80000-2.794319455)<449613
 assert n.release_CPU(529610.205680545,neural_paid=2.794319455)==449613
 assert n.release_CPU(348694.054558454,neural_paid=2.794319455,physical_paid=303.151122091)==175100

@pytest.mark.parametrize('kwargs',[dict(reserved=1,neural_paid=2),dict(reserved=90000,neural_paid=80001),dict(reserved=200000,neural_paid=0,physical_paid=93901),dict(reserved=100000,neural_paid=-1)])
def test_invalid_release_cannot_add_credit_or_cross_original_envelope(kwargs):
 with pytest.raises((PermissionError,ValueError)):n.release_CPU(**kwargs)


def test_whole_actual_terminal_preserves_cpu_debits_and_original_future():
 root=Path(__file__).resolve().parents[2].parent/'private/minimal-adaptation-20261027';files=sorted(root.glob('CORE_FAILED09_TERMINAL_READONLY_*private.json'))
 if not files:pytest.skip('private C09 actual terminal capture absent')
 p=json.loads(files[-1].read_bytes())['value'];b={k:base64.b64decode(v['base64'])for k,v in p['files'].items()};g=json.loads(b['GLOBAL']);v=n.propose(b['GLOBAL'],b['LOCAL'],request_id=p['request_id'],native_proof=p['native_proof'],effects=p['effects'],physical_bytes=b['PHYSICAL_LEDGER'],delivery_receipt_bytes=b['DELIVERY'],now_epoch=p['observed_epoch']+.1);after=v['proposed_global'];c.checked(after);c.paired(after,v['proposed_LOCAL'])
 assert v['receipt']['GPU_s']==691 and v['receipt']['neural_CPU_s']==4 and v['receipt']['physical_CPU_s']==303.151122091
 assert after['used']['H200_s']==g['used']['H200_s']+691
 assert after['CPU_s']==g['CPU_s']+4-c.progress(g[c.FIELD][p['request_id']])[2]
 assert after['PBS_submissions']==g['PBS_submissions']and after['new_storage_peak_bytes']==g['new_storage_peak_bytes']
 assert after['CPU_reserved_s']==449613.0
