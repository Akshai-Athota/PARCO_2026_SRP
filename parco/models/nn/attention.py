"""SRP Idea 2: tanh clipping applied to the encoder's attention logits
(QK^T / sqrt(d), before softmax), not just the decoder's action logits.

rl4co's `MultiHeadAttention` (parco/models/nn/transformer.py wraps it)
defaults to PyTorch's fused `scaled_dot_product_attention` kernel, which
never materializes the raw attention scores as a tensor -- there is nothing
to clip. To insert clipping we must use the "simple" (non-fused) SDPA path
and add the clip there, so this replicates
`rl4co.models.nn.attention.scaled_dot_product_attention_simple` exactly,
with a tanh clip inserted between the raw scores and the mask/softmax.
"""

import torch
import torch.nn.functional as F

from ..utils import tanh_clip


def make_clipped_sdpa_fn(tanh_clipping: float = 0.0, clip_mode: str = "scaled"):
    """Build an sdpa_fn (drop-in replacement for rl4co's `sdpa_fn` argument)
    that tanh-clips the raw attention scores before softmax.

    Args:
        tanh_clipping: clip value C. 0 disables clipping (plain SDPA).
        clip_mode: "fixed" (C*tanh(z)) or "scaled" (C*tanh(z/C)).
    """

    def clipped_sdpa_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        if is_causal and attn_mask is not None:
            raise ValueError("Cannot set both is_causal and attn_mask")

        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

        if tanh_clipping:
            scores = tanh_clip(scores, tanh_clipping, clip_mode)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if is_causal:
            s, l_ = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones((s, l_), device=scores.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.bool(), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        return torch.matmul(attn_weights, v)

    return clipped_sdpa_fn
