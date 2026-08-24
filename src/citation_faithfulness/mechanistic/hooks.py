from __future__ import annotations

import torch


def replace_position(clean: torch.Tensor, position: int):
    def hook(value: torch.Tensor, hook) -> torch.Tensor:
        result = value.clone(); result[:, position] = clean[:, position]; return result
    return hook

def replace_head_positions(source: torch.Tensor, head: int, positions: list[int]):
    def hook(value: torch.Tensor, hook) -> torch.Tensor:
        result = value.clone(); result[:, positions, head, :] = source[:, positions, head, :]; return result
    return hook

def scale_components(alpha: float, heads: set[tuple[int, int]] | None = None):
    heads = heads or set()
    def hook(value: torch.Tensor, hook) -> torch.Tensor:
        layer = int(hook.name.split(".")[1])
        result = value.clone()
        for selected_layer, head in heads:
            if selected_layer == layer: result[:, :, head, :] *= alpha
        return result
    return hook


def scale_output(alpha: float):
    def hook(value: torch.Tensor, hook) -> torch.Tensor:
        return value * alpha

    return hook
