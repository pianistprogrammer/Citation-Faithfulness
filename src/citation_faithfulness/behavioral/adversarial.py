def inject_phrase(documents: list[str], index: int, phrase: str) -> list[str]:
    if not 1 <= index <= len(documents): raise IndexError("document index is one-based and out of range")
    result = list(documents)
    result[index - 1] = f"{result[index - 1]}\n\n{phrase}"
    return result
