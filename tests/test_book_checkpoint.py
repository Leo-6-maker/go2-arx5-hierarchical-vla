import pytest
from scripts.pi0294.multitask.book_checkpoint import validate_missing,LM_HEAD


def test_only_proven_saved_language_embedding_alias_can_be_omitted():
    embed='backbone.model.model.embed_tokens.weight';keys={embed,LM_HEAD,'action_head.weight'}
    assert validate_missing(keys,keys-{LM_HEAD},[embed])=={LM_HEAD}
    with pytest.raises(ValueError):validate_missing(keys,keys-{LM_HEAD},[])
    with pytest.raises(ValueError):validate_missing(keys,keys-{'action_head.weight'},[embed])
    with pytest.raises(ValueError):validate_missing(keys,keys|{'random.weight'},[embed])
    assert validate_missing(keys,keys,[])==set()
