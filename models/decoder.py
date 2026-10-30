"""Progressive skip-connected decoder with ordered frequency refinement."""
import torch.nn.functional as F
import torch
from torch import nn
from .attention import (ChannelQueriedSelfAttention, SpaceQueriedSelfAttention,
                        WaveletQueriedSelfAttention)


class FrequencyRefinementBlock(nn.Module):
    """Figure 5 refinement sequence; the paper's preferred order is default."""
    ORDER = ("channel", "space", "wavelet")
    def __init__(self, channels, order=ORDER, heads=4, max_spatial_tokens=1024,
                 query_combine="add", use_task_queries=True,use_frequency_decoder=True):
        super().__init__()
        modules = {
            "channel": ChannelQueriedSelfAttention,
            "space": SpaceQueriedSelfAttention,
            "wavelet": WaveletQueriedSelfAttention,
        }
        if set(order) != set(self.ORDER) or len(order) != 3:
            raise ValueError("order must contain channel, space, and wavelet exactly once")
        self.order = tuple(order)
        self.blocks = nn.ModuleList([
            modules[name](channels, heads=heads, max_spatial_tokens=max_spatial_tokens,
                          query_combine=query_combine,use_task_query=use_task_queries)
            for name in self.order
        ])
        self.use_frequency_decoder=use_frequency_decoder

    def forward(self, x):
        if not self.use_frequency_decoder: return x
        for block in self.blocks:
            x = block(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention_order,
                 heads, max_spatial_tokens, query_combine,use_task_queries,
                 use_frequency_decoder):
        super().__init__()
        self.project = nn.Conv2d(in_channels+skip_channels, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU(),
            FrequencyRefinementBlock(out_channels, attention_order, heads, max_spatial_tokens,
                                     query_combine,use_task_queries,use_frequency_decoder))

    def forward(self, x, skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        return self.refine(self.project(torch.cat((x,skip),dim=1)))


class HierarchicalDecoder(nn.Module):
    def __init__(self, channels=(32,64,128), attention_order=FrequencyRefinementBlock.ORDER,
                 heads=4, max_spatial_tokens=1024, query_combine="add",
                 use_task_queries=True,use_frequency_decoder=True):
        super().__init__()
        stages=[]
        for i in range(len(channels)-2,-1,-1):
            stages.append(DecoderStage(channels[i+1],channels[i],channels[i],
                                       attention_order,heads,max_spatial_tokens,query_combine,
                                       use_task_queries,use_frequency_decoder))
        self.stages=nn.ModuleList(stages)

    def forward(self, features):
        x=features[-1]
        for stage, skip in zip(self.stages,reversed(features[:-1])):
            x=stage(x,skip)
        return x
