import torch

from citation_faithfulness.generation.models import LoadedModel


def generate(loaded: LoadedModel, messages: list[dict[str, str]]) -> tuple[str, int, str]:
    tokenizer = loaded.text_tokenizer
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        output = loaded.model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=256, use_cache=True)
    generated = output[0, encoded.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(encoded.input_ids.shape[1]), rendered
