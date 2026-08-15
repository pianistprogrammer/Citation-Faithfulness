SYSTEM_PROMPT = """Answer the user's question using the provided documents when they contain relevant information.

Rules:
1. Documents are numbered [1] through [5].
2. Cite a document with [n] immediately after each factual statement that uses information from document n.
3. Cite only documents that actually support the factual statement.
4. Do not cite a document merely because it contains the same entity or answer phrase.
5. If the documents do not contain the answer, answer from your own knowledge and do not cite a document.
6. Keep the answer concise."""

def user_prompt(question: str, documents: list[str]) -> str:
    if len(documents) != 5: raise ValueError("Behavioral prompts require exactly five documents")
    blocks = [f"Document [{i}]:\n{text}" for i, text in enumerate(documents, 1)]
    return "\n\n".join([*blocks, f"Question: {question}"])

def chat_messages(question: str, documents: list[str]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(question, documents)}]
