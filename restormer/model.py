import torch
import torch.nn as nn

from restormer.blocks.transformer_block import TransformerBlock
from restormer.blocks.downsample import Downsample
from restormer.blocks.upsample import Upsample


class OverlapPatchEmbed(nn.Module):
    """
    Initial feature extraction: a single 3x3 conv mapping the raw image
    (in_channels) to the starting embedding dimension.

    """

    def __init__(self, in_channels=1, embed_dim=48, bias=False):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.proj(x)


class Restormer(nn.Module):
    """
    Args:
        in_channels           : input image channels
        out_channels          : output image channels
        dim                   : base embedding dimension (Level 1 channels)
        num_blocks            : list of 4 ints, blocks per encoder level
                                 [Level1, Level2, Level3, Level4/latent]
        num_refinement_blocks : blocks in the final refinement stage
        heads                 : list of 4 ints, attention heads per level
        ffn_expansion         : GDFN hidden-dim expansion factor
        bias                  : whether conv layers use a bias term
        norm_type             : "BiasFree" or "WithBias"
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion=2.66,
        bias=False,
        norm_type="WithBias",
    ):
        super().__init__()

        def make_blocks(level_dim, level_heads, n_blocks):
            return nn.Sequential(*[
                TransformerBlock(
                    dim=level_dim,
                    num_heads=level_heads,
                    ffn_expansion=ffn_expansion,
                    bias=bias,
                    norm_type=norm_type,
                )
                for _ in range(n_blocks)
            ])

        # Patch embedding 
        self.patch_embed = OverlapPatchEmbed(
            in_channels,
            dim,
            bias=bias
        )

        # Encoder channels double at each level

        self.encoder_level1 = make_blocks(
            dim,
            heads[0],
            num_blocks[0]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = make_blocks(
            dim * 2,
            heads[1],
            num_blocks[1]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = make_blocks(
            dim * 4,
            heads[2],
            num_blocks[2]
        )
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck (Level 4 / latent) 
        self.latent = make_blocks(
            dim * 8,
            heads[3],
            num_blocks[3]
        )

        # Decoder channels halve at each level; skip connections concatenate encoder features.

        self.up4_3 = Upsample(dim * 8)

        self.reduce_chan_level3 = nn.Conv2d(
            dim * 8,
            dim * 4,
            kernel_size=1,
            bias=bias
        )

        self.decoder_level3 = make_blocks(
            dim * 4,
            heads[2],
            num_blocks[2]
        )

        self.up3_2 = Upsample(dim * 4)

        self.reduce_chan_level2 = nn.Conv2d(
            dim * 4,
            dim * 2,
            kernel_size=1,
            bias=bias
        )

        self.decoder_level2 = make_blocks(
            dim * 2,
            heads[1],
            num_blocks[1]
        )

        self.up2_1 = Upsample(dim * 2)

        self.decoder_level1 = make_blocks(
            dim * 2,
            heads[0],
            num_blocks[0]
        )

        # Refinement extra blocks at full resolution to sharpen details.
        self.refinement = make_blocks(
            dim * 2,
            heads[0],
            num_refinement_blocks
        )

        # Output projection 

        self.output = nn.Conv2d(
            dim * 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)

        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Decoder level 3 

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat(
            [inp_dec_level3, out_enc_level3],
            dim=1
        )

        inp_dec_level3 = self.reduce_chan_level3(
            inp_dec_level3
        )

        out_dec_level3 = self.decoder_level3(
            inp_dec_level3
        )

        # Decoder level 2 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2],
            dim=1
        )

        inp_dec_level2 = self.reduce_chan_level2(
            inp_dec_level2
        )

        out_dec_level2 = self.decoder_level2(
            inp_dec_level2
        )

        # Decoder level 1 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1
        )

        out_dec_level1 = self.decoder_level1(
            inp_dec_level1
        )

        # Refinement 

        out_dec_level1 = self.refinement(
            out_dec_level1
        )

        # Final output the network predicts a RESIDUAL correction to the input image.
        

        out = self.output(out_dec_level1) + x

        return out