from citation_faithfulness.behavioral.statements import extract_target_phrase, parse_cited_statements


def test_single_citation():
    item = parse_cited_statements("Paris is the capital of France [2].")[0]
    assert item.text == "Paris is the capital of France"
    assert item.citations == {2}

def test_multiple_citations():
    assert parse_cited_statements("Paris is in France [1][3].")[0].citations == {1, 3}


def test_extract_target_phrase():
    assert extract_target_phrase("The capital is New York City") == "New York City"
