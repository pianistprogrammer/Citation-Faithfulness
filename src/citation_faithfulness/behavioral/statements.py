import re, unicodedata
from dataclasses import dataclass
CITATION_RE = re.compile(r"\[([1-5])\]")
GROUP_RE = re.compile(r"((?:\[[1-5]\])+)")

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
