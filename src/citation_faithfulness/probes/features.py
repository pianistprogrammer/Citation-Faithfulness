import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def grouped_split(question_ids: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"question_id": sorted(set(question_ids))})
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    _, remainder_idx = next(first.split(frame, groups=frame.question_id))
    remainder = frame.iloc[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_local, test_local = next(second.split(remainder, groups=remainder.question_id))
    frame["split"] = "train"
    frame.loc[remainder.index[val_local], "split"] = "validation"
    frame.loc[remainder.index[test_local], "split"] = "test"
    return frame
