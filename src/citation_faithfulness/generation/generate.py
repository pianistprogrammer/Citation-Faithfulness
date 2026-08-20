import torch

from citation_faithfulness.generation.models import LoadedModel


def _generated_tokens(sequence: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    input_length = input_ids.shape[1]
    if sequence.shape[0] >= input_length and torch.equal(sequence[:input_length], input_ids[0]):
        return sequence[input_length:]
    return sequence


def generate(loaded: LoadedModel, messages: list[dict[str, str]], max_new_tokens: int = 256) -> tuple[str, int, str]:
    tokenizer = loaded.text_tokenizer
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(loaded.device)
    with torch.inference_mode():
        output = loaded.model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True)
    generated = _generated_tokens(output[0], encoded.input_ids)
    return tokenizer.decode(generated, skip_special_tokens=True), int(encoded.input_ids.shape[1]), rendered
