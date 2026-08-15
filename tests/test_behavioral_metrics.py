from citation_faithfulness.metrics.behavioral import aggregate, wilson


def test_behavioral_denominators_exclude_unavailable_and_indeterminate():
    rows = [
        {"model_id": "m", "condition": "random", "condition_available": True, "statement_recovered": True, "adversarial_doc_cited": True},
        {"model_id": "m", "condition": "random", "condition_available": True, "statement_recovered": False, "adversarial_doc_cited": False},
        {"model_id": "m", "condition": "random", "condition_available": False, "statement_recovered": None, "adversarial_doc_cited": None},
    ]
    metric = aggregate(rows)[0]
    assert metric["condition_available_count"] == 2
    assert metric["statement_recovery_rate"] == 0.5
    assert metric["post_rationalization_rate_conditional"] == 1.0


def test_wilson_interval_is_bounded():
    low, high = wilson(5, 10)
    assert 0 < low < 0.5 < high < 1
