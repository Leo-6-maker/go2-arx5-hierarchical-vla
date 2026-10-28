from copy import deepcopy
import pytest
from scripts.pi0294.generalization.minimal_native_audit import match_saved_effects


def test_read_time_changes_without_changing_the_effects():
    saved=dict(finished_epoch=10,model_load_attempts=0,query_attempts=0,
               pending_updates=[],unknown_query_intents=[])
    current=dict(saved,finished_epoch=11)
    assert match_saved_effects(current,saved)==saved
    assert current['finished_epoch']==11


@pytest.mark.parametrize('change', ['model','pending','unknown','earlier'])
def test_changed_effect_or_unknown_still_rejected(change):
    saved=dict(finished_epoch=10,model_load_attempts=0,query_attempts=0,
               pending_updates=[],unknown_query_intents=[])
    current=deepcopy(saved);current['finished_epoch']=11
    if change=='model':current['model_load_attempts']=1
    if change=='pending':current['pending_updates']=['unresolved']
    if change=='unknown':current['unknown_query_intents']=['unresolved']
    if change=='earlier':current['finished_epoch']=9
    with pytest.raises(PermissionError):match_saved_effects(current,saved)
