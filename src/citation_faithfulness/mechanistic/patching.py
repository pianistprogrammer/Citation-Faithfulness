def normalized_recovery(clean: float, corrupted: float, patched: float) -> float:
    denominator = clean - corrupted
    if denominator == 0: raise ValueError("clean and corrupted logit differences must differ")
    return (patched - corrupted) / denominator
