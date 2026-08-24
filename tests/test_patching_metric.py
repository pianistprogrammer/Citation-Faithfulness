from types import SimpleNamespace

import torch

from citation_faithfulness.mechanistic.hooks import replace_head_positions, replace_position, scale_components, scale_output
from citation_faithfulness.mechanistic.patching import normalized_recovery


def test_normalized_recovery(): assert normalized_recovery(4, -2, 1) == 0.5


def test_transformer_lens_hook_keyword_signature():
    value = torch.zeros((1, 3, 2))
    clean = torch.ones((1, 3, 2))
    assert replace_position(clean, 1)(value, hook=object())[0, 1].tolist() == [1, 1]

    heads = torch.zeros((1, 3, 2, 2))
    source = torch.ones((1, 3, 2, 2))
    assert replace_head_positions(source, 1, [0, 2])(heads, hook=object())[0, 0, 1].tolist() == [1, 1]

    assert scale_components(2.0, {(4, 1)})(source, hook=SimpleNamespace(name="blocks.4.attn.hook_z"))[0, 0, 1].tolist() == [2, 2]
    assert scale_output(3.0)(clean, hook=object())[0, 0].tolist() == [3, 3]
