import torch.nn as nn

from restormer.blocks.layer_norm import LayerNorm
from restormer.blocks.mdta import MDTA
from restormer.blocks.gdfn import GDFN


class TransformerBlock(nn.Module):
    """
    One Restormer Transformer Block, combining channel attention (MDTA)
    and gated feedforward (GDFN), each wrapped in its own pre-norm +
    residual connection:

        X_hat = X + MDTA(Norm1(X))
        X_out = X_hat + GDFN(Norm2(X_hat))

    This is the "pre-norm" transformer design (normalize BEFORE the
    sub-layer, not after) - same convention used in most modern
    transformers, chosen because it stabilizes training in deep stacks.

    Args:
        dim               : number of channels C (unchanged in/out)
        num_heads         : number of attention heads for MDTA
        ffn_expansion     : hidden-dim expansion factor for GDFN
        bias              : whether conv layers use a bias term
        norm_type         : "BiasFree" or "WithBias", passed to LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion=2.66,
        bias=False,
        norm_type="WithBias"
    ):
        super().__init__()

        self.norm1 = LayerNorm(
            dim,
            norm_type=norm_type
        )

        self.attn = MDTA(
            dim,
            num_heads=num_heads,
            bias=bias
        )

        self.norm2 = LayerNorm(
            dim,
            norm_type=norm_type
        )

        self.ffn = GDFN(
            dim,
            ffn_expansion=ffn_expansion,
            bias=bias
        )

    def forward(self, x):
        # Attention sub-layer: pre-norm + residual
        x = x + self.attn(self.norm1(x))

        # Feedforward sub-layer: pre-norm + residual
        x = x + self.ffn(self.norm2(x))

        return x