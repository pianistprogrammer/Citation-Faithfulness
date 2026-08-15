import re
import unicodedata
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[([1-5])\]")
GROUP_RE = re.compile(r"((?:\[[1-5]\])+)")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"}

@dataclass(frozen=True)
class CitedStatement:
    text: str
    citations: frozenset[int]

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = CITATION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(".,;:!?()[]{}\"' ")

def parse_cited_statements(answer: str) -> list[CitedStatement]:
    result = []
    for match in GROUP_RE.finditer(answer):
        prefix = answer[:match.start()].rstrip()
        boundary = max(prefix.rfind("."), prefix.rfind("?"), prefix.rfind("!"), prefix.rfind("\n"))
        statement = prefix[boundary + 1:].strip()
        citations = frozenset(int(x) for x in CITATION_RE.findall(match.group()))
        if statement: result.append(CitedStatement(statement, citations))
    return result

def sentence_with_phrase(answer: str, phrase: str) -> str | None:
    wanted = normalize(phrase)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if wanted and wanted in normalize(sentence): return sentence
    return None


def extract_target_phrase(statement: str) -> str | None:
    """Extract the maximal trailing content span used by the fixed intervention.

    The PRD's noun/proper-noun/number rule is implemented conservatively without
    adding an unapproved POS package: punctuation and stopwords terminate the
    trailing candidate, while capitalized, numeric, and content tokens extend it.
    """
    tokens = re.findall(r"[\w'-]+", unicodedata.normalize("NFKC", statement), re.UNICODE)
    selected: list[str] = []
    for token in reversed(tokens):
        if token.lower() in STOPWORDS:
            if selected:
                break
            continue
        selected.append(token)
        if len(selected) == 8:
            break
    if not selected:
        return None
    return " ".join(reversed(selected))
