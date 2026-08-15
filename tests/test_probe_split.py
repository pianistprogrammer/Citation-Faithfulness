from citation_faithfulness.probes.features import grouped_split

def test_no_question_leakage():
    result = grouped_split([str(i // 3) for i in range(300)])
    assert result.question_id.nunique() == len(result)
    assert set(result.split) == {"train", "validation", "test"}
