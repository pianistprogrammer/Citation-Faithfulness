import pytest
from citation_faithfulness.generation.prompts import user_prompt

def test_prompt_has_five_documents():
    prompt = user_prompt("Where?", ["a", "b", "c", "d", "e"])
    assert "Document [5]:\ne" in prompt and prompt.endswith("Question: Where?")

def test_prompt_rejects_wrong_count():
    with pytest.raises(ValueError): user_prompt("Where?", ["a"])
