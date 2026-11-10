"""Independent AIRFormer-inspired reconstruction, not an exact paper model.

See docs/architecture/README.md for confirmed graph corrections and assumptions.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from .patch_embed import PatchEmbed
from .encoder import HierarchicalEncoder
from .supplementary_prior import SupplementaryPrior
from .decoder import HierarchicalDecoder


class AIRFormer(nn.Module):
    def __init__(self, base_channels=32, stage_blocks=(2,2,2,2),
                 patch_size=3, patch_stride=1, patch_padding=1,
                 attention_heads=4, max_spatial_tokens=1024,
                 attention_order=("channel","space","wavelet"),
                 query_combine="add",
                 encoder_token_mixer="pooling",decoder_mode=None,
                 use_supplementary_prior=True, use_task_queries=True,
                 use_frequency_decoder=None, residual_image=True,
                 gradient_checkpointing=False, spatial_kv_pooling=False):
        super().__init__()
        if len(stage_blocks) != 4 or any(int(d) != d or d < 1 for d in stage_blocks):
            raise ValueError("stage_blocks must contain four positive integer depths")
        channels=tuple(base_channels*2**i for i in range(4))
        self.patch_embed=PatchEmbed(3,channels[0],patch_size,patch_stride,patch_padding)
        self.encoder=HierarchicalEncoder(channels,stage_blocks,encoder_token_mixer,
                                         attention_heads,max_spatial_tokens)
        self.use_supplementary_prior=use_supplementary_prior
        if use_supplementary_prior:
            self.supplementary_priors=nn.ModuleList([
                SupplementaryPrior(c, c, output_channels=channels[i+1])
                for i,c in enumerate(channels[:-1])])
        if decoder_mode is None:
            decoder_mode="frequency" if use_frequency_decoder is not False else "none"
        self.decoder=HierarchicalDecoder(channels,attention_order,attention_heads,
                                         max_spatial_tokens,query_combine,use_task_queries,
                                         decoder_mode, spatial_kv_pooling)
        # A07: a single RGB projection after assumed skip-fusion stages. The paper
        # labels a hierarchical reconstruction tail but does not specify this one.
        self.reconstruct=nn.Conv2d(channels[0],3,3,padding=1)
        self.residual_image=residual_image
        self.gradient_checkpointing=gradient_checkpointing

    @staticmethod
    def _pad(x):
        h,w=x.shape[-2:]
        ph,pw=(-h)%8,(-w)%8
        if ph or pw:
            mode="reflect" if h>ph and w>pw and h>1 and w>1 else "replicate"
            x=F.pad(x,(0,pw,0,ph),mode=mode)
        return x,(h,w)

    def forward(self, degraded_image):
        # Four encoder positions, three recurrent priors injected pre-block.
        # Padding here serves this chosen hierarchy; utils/inference.py applies
        # the separately documented multiple-of-128 evaluation policy.
        image,size=self._pad(degraded_image)
        x=image
        x=self.patch_embed(x)
        features, _ = self.encoder(x,self.supplementary_priors if self.use_supplementary_prior else None)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            decoded=checkpoint(lambda *fs:self.decoder(fs),*features,use_reentrant=False)
        else:
            decoded=self.decoder(features)
        decoded=F.interpolate(decoded,size=image.shape[-2:],mode="bilinear",align_corners=False)
        restored=self.reconstruct(decoded)
        if self.residual_image:
            # A07: global image residual is an adaptation, not confirmed by Fig. 4.
            restored=restored+image[:,:3]
        h,w=size
        return restored[...,:h,:w].clamp(0,1)
