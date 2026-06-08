"""MambaMPD: a Mamba-driven segmentation framework for marine pollution detection.

Assembles the full model described in the paper (Fig. 2):

    Input -> FAA (WT block) -> stem -> VSS encoder
          -> SE-ResDecoder (UnetrUpBlock + SE) with deep supervision,
             whose skip connections are refined by Edge-Guided Attention (EGA).

Compared with the original research code, this implementation:

* registers FAA, EGA and the SE blocks as proper sub-modules (they are created
  once in ``__init__`` and trained, instead of being re-instantiated inside
  ``forward`` with fresh random weights);
* is device-agnostic (no hard-coded ``.cuda()`` calls); and
* exposes a clean ``deep_supervision`` switch matching the paper's training
  objective.
"""

import re

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .faa import FAA, SELayer
from .ega import EGA
from .vss_encoder import VSSMEncoder


class MambaMPD(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=5,
        feat_size=(48, 96, 192, 384, 768),
        hidden_size=768,
        norm_name="instance",
        res_block=True,
        spatial_dims=2,
        deep_supervision=True,
        use_faa=True,
        use_ega=True,
        faa_levels=2,
        faa_wavelet="db1",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.feat_size = list(feat_size)
        self.hidden_size = hidden_size
        self.deep_supervision = deep_supervision
        self.use_faa = use_faa
        self.use_ega = use_ega

        # Frequency-Aware Augmentation on the raw input (before patch embedding).
        self.faa = FAA(channels=in_chans, wt_levels=faa_levels, wt_type=faa_wavelet) if use_faa else None

        # Stem: 7x7 stride-2 conv + instance norm -> feat_size[0] channels at H/2.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, self.feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(self.feat_size[0], eps=1e-5, affine=True),
        )

        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=self.feat_size[0])

        # Channel-aligning encoder blocks for the skip connections.
        self.encoder1 = UnetrBasicBlock(spatial_dims, in_chans, self.feat_size[0], 3, 1, norm_name=norm_name, res_block=res_block)
        self.encoder2 = UnetrBasicBlock(spatial_dims, self.feat_size[0], self.feat_size[1], 3, 1, norm_name=norm_name, res_block=res_block)
        self.encoder3 = UnetrBasicBlock(spatial_dims, self.feat_size[1], self.feat_size[2], 3, 1, norm_name=norm_name, res_block=res_block)
        self.encoder4 = UnetrBasicBlock(spatial_dims, self.feat_size[2], self.feat_size[3], 3, 1, norm_name=norm_name, res_block=res_block)
        self.encoder5 = UnetrBasicBlock(spatial_dims, self.feat_size[3], self.feat_size[4], 3, 1, norm_name=norm_name, res_block=res_block)

        # SE-ResDecoder: UnetrUpBlock followed by an SE block at each stage.
        self.decoder6 = UnetrUpBlock(spatial_dims, hidden_size, self.feat_size[4], 3, 2, norm_name=norm_name, res_block=res_block)
        self.decoder5 = UnetrUpBlock(spatial_dims, hidden_size, self.feat_size[3], 3, 2, norm_name=norm_name, res_block=res_block)
        self.decoder4 = UnetrUpBlock(spatial_dims, self.feat_size[3], self.feat_size[2], 3, 2, norm_name=norm_name, res_block=res_block)
        self.decoder3 = UnetrUpBlock(spatial_dims, self.feat_size[2], self.feat_size[1], 3, 2, norm_name=norm_name, res_block=res_block)
        self.decoder2 = UnetrUpBlock(spatial_dims, self.feat_size[1], self.feat_size[0], 3, 2, norm_name=norm_name, res_block=res_block)
        self.decoder1 = UnetrBasicBlock(spatial_dims, self.feat_size[0], self.feat_size[0], 3, 1, norm_name=norm_name, res_block=res_block)

        self.se6 = SELayer(self.feat_size[4])
        self.se5 = SELayer(self.feat_size[3])
        self.se4 = SELayer(self.feat_size[2])
        self.se3 = SELayer(self.feat_size[1])
        self.se2 = SELayer(self.feat_size[0])

        # Edge-Guided Attention on the three mid-level skip features.
        if use_ega:
            self.ega4 = EGA(self.feat_size[3])  # 384
            self.ega3 = EGA(self.feat_size[2])  # 192
            self.ega2 = EGA(self.feat_size[1])  # 96
            # Single-channel edge-guidance heads from the coarser decoder stages.
            self.edge_head4 = nn.Conv2d(self.feat_size[4], 1, kernel_size=1)
            self.edge_head3 = nn.Conv2d(self.feat_size[3], 1, kernel_size=1)
            self.edge_head2 = nn.Conv2d(self.feat_size[2], 1, kernel_size=1)

        # Deep-supervision heads (multi-class) on the four finest decoder features.
        self.out_layers = nn.ModuleList(
            [UnetOutBlock(spatial_dims, in_channels=self.feat_size[i], out_channels=out_chans) for i in range(4)]
        )

    def forward(self, x_in):
        x = self.faa(x_in) if self.use_faa else x_in
        x1 = self.stem(x)
        vss_outs = self.vssm_encoder(x1)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(vss_outs[0])
        enc3 = self.encoder3(vss_outs[1])
        enc4 = self.encoder4(vss_outs[2])
        enc5 = self.encoder5(vss_outs[3])
        enc_hidden = vss_outs[4]

        dec4 = self.se6(self.decoder6(enc_hidden, enc5))
        if self.use_ega:
            aux4 = torch.sigmoid(self.edge_head4(dec4))
            enc4 = self.ega4(enc4, aux4)
        dec3 = self.se5(self.decoder5(dec4, enc4))

        if self.use_ega:
            aux3 = torch.sigmoid(self.edge_head3(dec3))
            enc3 = self.ega3(enc3, aux3)
        dec2 = self.se4(self.decoder4(dec3, enc3))

        if self.use_ega:
            aux2 = torch.sigmoid(self.edge_head2(dec2))
            enc2 = self.ega2(enc2, aux2)
        dec1 = self.se3(self.decoder3(dec2, enc2))

        dec0 = self.se2(self.decoder2(dec1, enc1))
        dec_out = self.decoder1(dec0)

        if self.deep_supervision:
            feat_out = [dec_out, dec1, dec2, dec3]
            return [self.out_layers[i](feat_out[i]) for i in range(4)]
        return self.out_layers[0](dec_out)

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def load_pretrained_ckpt(model, ckpt_path):
    """Load ImageNet-pretrained VMamba-Tiny weights into the VSS encoder.

    The patch-embed / classification-head / final-norm parameters are skipped so
    the segmentation model keeps its own (re-)initialised layers, matching the
    paper's protocol ("all other encoder layers keep their ImageNet-pretrained
    weights").
    """
    print(f"Loading pretrained encoder weights from: {ckpt_path}")
    skip_params = {
        "norm.weight", "norm.bias", "head.weight", "head.bias",
        "patch_embed.proj.weight", "patch_embed.proj.bias",
        "patch_embed.norm.weight", "patch_embed.norm.bias",
    }

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model_dict = model.state_dict()
    for k, v in state.items():
        if k in skip_params:
            continue
        kr = f"vssm_encoder.{k}"
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
        if kr in model_dict and v.shape == model_dict[kr].shape:
            model_dict[kr] = v
    model.load_state_dict(model_dict)
    return model


def build_mambampd(num_classes=5, in_chans=3, deep_supervision=True, use_faa=True, use_ega=True, **kwargs):
    """Convenience constructor for the full MambaMPD model."""
    return MambaMPD(
        in_chans=in_chans,
        out_chans=num_classes,
        deep_supervision=deep_supervision,
        use_faa=use_faa,
        use_ega=use_ega,
        **kwargs,
    )
