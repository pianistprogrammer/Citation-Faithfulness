import numpy as np

from citation_faithfulness.data.conflictbank import _cites_doc1, answer_contains, classify
from citation_faithfulness.mechanistic.patching import _bootstrap_mean_ci
from citation_faithfulness.probes.features import _locate_end


class WordTokenizer:
    """Whitespace tokenizer stub: token ids are 1-based indices into a vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 1))
        return ids


def test_answer_contains_recovers_verbose_parametric_answers():
    # Plain substring match fails here; content-token match recovers it.
    assert answer_contains("The capital city is Paris, in France.", "Paris")
    assert answer_contains("It is located in New York City.", "New York")
    assert not answer_contains("The answer is Berlin.", "Paris")


def test_answer_contains_ignores_stopword_only_candidates():
    assert not answer_contains("anything at all", "the of and")


def test_classify_three_way():
    assert classify("The capital is Paris.", "Paris", "Lyon") == "PARAMETRIC_MATCH"
    assert classify("The capital is Lyon.", "Paris", "Lyon") == "CONTEXT_MATCH"
    assert classify("Paris and Lyon are both mentioned.", "Paris", "Lyon") == "AMBIGUOUS"
    assert classify("No idea.", "Paris", "Lyon") == "AMBIGUOUS"


def test_cites_doc1():
    assert _cites_doc1("The capital is Lyon [1].")
    assert not _cites_doc1("The capital is Lyon.")
    assert not _cites_doc1("See document [2] for details.")


def test_probe_anchor_is_label_independent():
    tokenizer = WordTokenizer()
    answer_cited = "the capital of poland is warsaw [1] according to the source"
    answer_uncited = "the capital of poland is warsaw according to the source"
    phrase = "warsaw"
    ids_cited = tokenizer.encode(answer_cited)
    ids_uncited = tokenizer.encode(answer_uncited)
    end_cited = _locate_end(ids_cited, tokenizer, phrase)
    end_uncited = _locate_end(ids_uncited, tokenizer, phrase)
    # Anchor points at the target phrase token in both cases, before any citation.
    assert end_cited is not None and end_uncited is not None
    assert ids_cited[end_cited] == tokenizer.vocab["warsaw"]
    assert ids_uncited[end_uncited] == tokenizer.vocab["warsaw"]


def test_locate_end_returns_none_when_absent():
    tokenizer = WordTokenizer()
    ids = tokenizer.encode("no target phrase here")
    assert _locate_end(ids, tokenizer, "warsaw") is None


def test_bootstrap_mean_ci_is_ordered_and_deterministic():
    values = np.array([0.0, 0.05, 0.1, 0.02, 0.03, 0.04])
    mean_a, low_a, high_a = _bootstrap_mean_ci(values, resamples=500)
    mean_b, low_b, high_b = _bootstrap_mean_ci(values, resamples=500)
    assert (mean_a, low_a, high_a) == (mean_b, low_b, high_b)
    assert low_a <= mean_a <= high_a
