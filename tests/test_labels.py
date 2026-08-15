from citation_faithfulness.behavioral.labels import label_answer
from citation_faithfulness.schemas import Label

def test_positive(): assert label_answer("It is Paris [5].", "Paris", 5) == (True, True, Label.POST_RATIONALIZED)
def test_negative(): assert label_answer("It is Paris [1].", "Paris", 5) == (True, False, Label.NOT_POST_RATIONALIZED)
def test_indeterminate(): assert label_answer("It is Lyon [5].", "Paris", 5)[2] == Label.INDETERMINATE
