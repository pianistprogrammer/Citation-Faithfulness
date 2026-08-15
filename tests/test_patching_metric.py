from citation_faithfulness.mechanistic.patching import normalized_recovery

def test_normalized_recovery(): assert normalized_recovery(4, -2, 1) == 0.5
