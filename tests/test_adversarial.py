from citation_faithfulness.behavioral.adversarial import inject_phrase

def test_injection_changes_only_selected_document():
    original = ["a", "b", "c"]
    assert inject_phrase(original, 2, "Paris") == ["a", "b\n\nParis", "c"]
    assert original == ["a", "b", "c"]
