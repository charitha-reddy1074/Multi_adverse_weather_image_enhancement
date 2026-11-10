# Architecture to code

Shapes are NCHW with batch omitted unless stated. Dimensions below use the actual 128 x 128 training crop. Line ranges refer to the source hashes recorded in sources/source_inventory.csv.

## Model initialization

- Paper concept: Model initialization.
- Implemented symbol: `models/airformer.py::AIRFormer` (AST symbol lines 15-75).
- Input: 3 x H x W. Output: 3 x H x W.
- Behavior and assumptions: Builds four widths, three priors, skip decoder and RGB head. Widths/depths are experiment choices.

```python
28:         channels=tuple(base_channels*2**i for i in range(4))
29:         self.patch_embed=PatchEmbed(3,channels[0],patch_size,patch_stride,patch_padding)
30:         self.encoder=HierarchicalEncoder(channels,stage_blocks,encoder_token_mixer,
31:                                          attention_heads,max_spatial_tokens)
32:         self.use_supplementary_prior=use_supplementary_prior
33:         if use_supplementary_prior:
34:             self.supplementary_priors=nn.ModuleList([
35:                 SupplementaryPrior(c, c, output_channels=channels[i+1])
36:                 for i,c in enumerate(channels[:-1])])
37:         if decoder_mode is None:
38:             decoder_mode="frequency" if use_frequency_decoder is not False else "none"
39:         self.decoder=HierarchicalDecoder(channels,attention_order,attention_heads,
40:                                          max_spatial_tokens,query_combine,use_task_queries,
41:                                          decoder_mode, spatial_kv_pooling)
42:         # A07: a single RGB projection after assumed skip-fusion stages. The paper
43:         # labels a hierarchical reconstruction tail but does not specify this one.
```

## Patch embedding

- Paper concept: Patch embedding.
- Implemented symbol: `models/patch_embed.py::PatchEmbed` (AST symbol lines 5-19).
- Input: 3 x H x W. Output: 16 x H x W.
- Behavior and assumptions: Dense 3x3 stride-1 projection with GroupNorm, not disjoint image patches.

```python
5: class PatchEmbed(nn.Module):
6:     """Chosen dense 3x3 convolution + GroupNorm(1,C) patch projection.
7: 
8:     Figure 4 names patch embeddings at each level; its kernels, strides and
9:     feature widths are not specified. This class alone does not reproduce them.
10:     """
11:     def __init__(self, in_channels=3, embed_dim=32, patch_size=3, stride=1,
12:                  padding=1, norm=True):
13:         super().__init__()
14:         self.proj = nn.Conv2d(in_channels, embed_dim, patch_size,
15:                               stride=stride, padding=padding)
16:         self.norm = nn.GroupNorm(1, embed_dim) if norm else nn.Identity()
17: 
18:     def forward(self, x):
19:         return self.norm(self.proj(x))
```

## Encoder forward pass

- Paper concept: Encoder forward pass.
- Implemented symbol: `models/encoder.py::HierarchicalEncoder` (AST symbol lines 55-91).
- Input: 16 x H x W. Output: four feature maps.
- Behavior and assumptions: Stride-2 convolution and prior addition before each next stage. First SP gets zeros.

```python
55: class HierarchicalEncoder(nn.Module):
56:     def __init__(self, channels=(32, 64, 128, 256), depths=(2, 2, 2, 2),
57:                  token_mixer="pooling",attention_heads=4,max_spatial_tokens=1024):
58:         super().__init__()
59:         if len(channels) != 4 or len(depths) != 4:
60:             raise ValueError("Figure 4 requires four encoder positions")
61:         self.stages = nn.ModuleList([
62:             nn.Sequential(*(FrequencyGuidedBlock(c,token_mixer=token_mixer,
63:                 attention_heads=attention_heads,max_spatial_tokens=max_spatial_tokens)
64:                 for _ in range(d)))
65:             for c, d in zip(channels, depths)
66:         ])
67:         self.downsamples = nn.ModuleList([
68:             nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1)
69:             for i in range(len(channels)-1)
70:         ])
71: 
72:     def forward(self, x, supplementary_modules=None):
73:         features, priors = [], []
74:         previous = None
75:         for i, stage in enumerate(self.stages):
76:             if i:
77:                 preceding = x
78:                 x = self.downsamples[i-1](x)
79:                 if supplementary_modules is not None:
80:                     # Eq. 10: first SP sees zero feature and zero prior. Later
81:                     # SPs see the preceding enhanced stage and preceding SP.
82:                     source = torch.zeros_like(preceding) if i == 1 else preceding
83:                     prev = torch.zeros_like(preceding) if i == 1 else previous
84:                     previous = supplementary_modules[i-1](source, prev)
85:                     if previous.shape != x.shape:
86:                         raise RuntimeError("Supplementary prior and patch embedding must align")
```

## Supplementary Prior

- Paper concept: Supplementary Prior.
- Implemented symbol: `models/supplementary_prior.py::SupplementaryPrior` (AST symbol lines 34-80).
- Input: C x H x W plus prior. Output: 2C x H/2 x W/2.
- Behavior and assumptions: Haar bands, detail-only filtering, 6C fusion, refinement and stride-2 alignment are assumed internals.

```python
66: 
67:     def forward(self, encoder_feature, previous_prior=None):
68:         b, c, h, w = encoder_feature.shape
69:         # LL passes unchanged inside filter_wavelet_bands. The following fusion
70:         # can still change it; preservation here does not mean an identity prior.
```

## Wavelet decomposition

- Paper concept: Wavelet decomposition.
- Implemented symbol: `models/supplementary_prior.py::dwt2` (AST symbol lines 7-19).
- Input: C x H x W. Output: C x 4 x ceil(H/2) x ceil(W/2).
- Behavior and assumptions: Fixed orthonormal Haar; odd boundaries replicated. Family not specified by paper.

```python
7: def dwt2(x):
8:     """Orthonormal Haar transform. Returns coeffs [B,C,4,ceil(H/2),ceil(W/2)] and original size."""
9:     h, w = x.shape[-2:]
10:     # A08: fixed Haar basis and replicate extension are engineering choices.
11:     # Eq. (9) defines a general wavelet transform, not this specific family.
12:     ph, pw = h % 2, w % 2
13:     if ph or pw:
14:         x = F.pad(x, (0, pw, 0, ph), mode="replicate")
15:     a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
16:     c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
17:     bands = torch.stack(((a+b+c+d)*.5, (-a-b+c+d)*.5,
18:                          (-a+b-c+d)*.5, (a-b-c+d)*.5), dim=2)
19:     return bands, (h, w)
```

## Channel attention

- Paper concept: Channel attention.
- Implemented symbol: `models/attention.py::QueriedAttention` (AST symbol lines 10-117).
- Input: C x H x W. Output: C x H x W.
- Behavior and assumptions: Channel mode transposes to channel affinities; per-head matrix (C/2) x (C/2). Additive learned query is assumed.

```python
77:         if self.mode == "channel":
78:             q = F.normalize(q.transpose(-1, -2), dim=-1)
79:             k = F.normalize(k.transpose(-1, -2), dim=-1)
80:             v = v.transpose(-1, -2)
81:         if self.use_task_query:
82:             q = q * (1 + torch.tanh(self.task_query)) if self.query_combine == "gate" else q + self.task_query
83:         self.last_attention_shape = (*q.shape[:-1], k.shape[-2])
84:         # SDPA computes full attention with softmax over keys without requiring
85:         # a materialized HW-by-HW matrix on supported CUDA backends.
86:         def attend(query, key, value):
87:             # Channel attention uses HW as the embedding dimension. Full-size
```

## Spatial attention

- Paper concept: Spatial attention.
- Implemented symbol: `models/attention.py::QueriedAttention` (AST symbol lines 10-117).
- Input: C x H x W. Output: C x H x W.
- Behavior and assumptions: Spatial queries retain HW positions; K/V adaptively pooled to at most 16x16 in trained configuration.

```python
71:         if self.mode != "channel" and self.spatial_kv_pooling and kv_in.shape[-2] * kv_in.shape[-1] > self.max_tokens:
72:             side = max(1, int(self.max_tokens**.5))
73:             kv_in = F.adaptive_avg_pool2d(kv_in, (side, side))
74:         q = self._split(self.q_proj(q_in))
75:         k = self._split(self.k_proj(kv_in))
76:         v = self._split(self.v_proj(kv_in))
77:         if self.mode == "channel":
```

## Wavelet attention

- Paper concept: Wavelet attention.
- Implemented symbol: `models/attention.py::QueriedAttention` (AST symbol lines 10-117).
- Input: C x H x W. Output: C x H x W.
- Behavior and assumptions: Normalize before DWT. Project packed 4C to C at H/2,W/2; K/V from this. Separate IDWT produces C/4 spatial channels.

```python
66:         if self.mode == "wavelet":
67:             coeffs, size = dwt2(q_in)
68:             hh, ww = coeffs.shape[-2:]
69:             kv_in = self.wavelet_proj(coeffs.reshape(b, 4*c, hh, ww))
70:             inverse = idwt2(kv_in.reshape(b, c//4, 4, hh, ww), size)
71:         if self.mode != "channel" and self.spatial_kv_pooling and kv_in.shape[-2] * kv_in.shape[-1] > self.max_tokens:
```

## Decoder forward pass

- Paper concept: Decoder forward pass.
- Implemented symbol: `models/decoder.py::HierarchicalDecoder` (AST symbol lines 65-81).
- Input: 128 x 16 x 16 + skips. Output: 16 x 128 x 128.
- Behavior and assumptions: Three bilinear upsample/concat/project stages; each applies Channel, Space, Wavelet.

```python
65: class HierarchicalDecoder(nn.Module):
66:     def __init__(self, channels=(32,64,128,256), attention_order=FrequencyRefinementBlock.ORDER,
67:                  heads=4, max_spatial_tokens=1024, query_combine="add",
68:                  use_task_queries=True,decoder_mode="frequency",spatial_kv_pooling=False):
69:         super().__init__()
70:         stages=[]
71:         for i in range(len(channels)-2,-1,-1):
72:             stages.append(DecoderStage(channels[i+1],channels[i],channels[i],
73:                                        attention_order,heads,max_spatial_tokens,query_combine,
74:                                        use_task_queries,decoder_mode,spatial_kv_pooling))
75:         self.stages=nn.ModuleList(stages)
76: 
77:     def forward(self, features):
78:         x=features[-1]
79:         for stage, skip in zip(self.stages,reversed(features[:-1])):
80:             x=stage(x,skip)
81:         return x
```

## Reconstruction

- Paper concept: Reconstruction.
- Implemented symbol: `models/airformer.py::forward` (AST symbol lines 57-75).
- Input: 16 x H x W. Output: 3 x H x W.
- Behavior and assumptions: RGB 3x3 convolution, input residual, original crop and clamp. This tail is an adaptation.

```python
70:         restored=self.reconstruct(decoded)
71:         if self.residual_image:
72:             # A07: global image residual is an adaptation, not confirmed by Fig. 4.
73:             restored=restored+image[:,:3]
74:         h,w=size
75:         return restored[...,:h,:w].clamp(0,1)
```

## Multi-domain loss

- Paper concept: Multi-domain loss.
- Implemented symbol: `losses/multidomain.py::MultiDomainConsistencyLoss` (AST symbol lines 32-55).
- Input: prediction + target: B x 3 x H x W. Output: scalar + detached parts.
- Behavior and assumptions: Elementwise Smooth L1 mean plus .04 smoothed per-image DCT norm; differs from literal Eq.12.

```python
32: class MultiDomainConsistencyLoss(nn.Module):
33:     """L_smooth1 + lambda_freq * sqrt(||DCT(p)-DCT(t)||_2^2 + eps^2).
34: 
35:     Reduction policy: PyTorch elementwise Smooth L1 mean across B,C,H,W;
36:     frequency coefficients use one Frobenius norm per image across C,H,W,
37:     followed by a mean over batch. A09: printed Eq. (12) defines E as a spatial
38:     mean residual *before* the piecewise penalty; this implementation averages
39:     elementwise penalties instead. They differ on cancellation examples. The
40:     conventional choice is retained, but must not be claimed literally exact.
41:     """
42:     def __init__(self,lambda_freq=0.04,eps=0.001):
43:         super().__init__()
44:         self.lambda_freq=float(lambda_freq)
45:         self.eps=float(eps)
46: 
47:     def forward(self,pred,target):
48:         if pred.shape!=target.shape or pred.ndim!=4:
49:             raise ValueError("pred and target must have equal NCHW shapes")
50:         # A09: mean SmoothL1(error), not SmoothL1(mean(error)) in printed Eq. 12.
51:         spatial=F.smooth_l1_loss(pred.float(),target.float(),reduction="mean")
52:         difference=(dct2(pred)-dct2(target)).flatten(1)
53:         frequency=torch.sqrt(difference.square().sum(dim=1)+self.eps**2).mean()
54:         total=spatial+self.lambda_freq*frequency
55:         return total,{"smooth_l1":spatial.detach(),"dct_frequency":frequency.detach()}
```

## Training step

- Paper concept: Training step.
- Implemented symbol: `scripts/train.py::main` (AST symbol lines 45-234).
- Input: B=8, 3 x 128 x 128. Output: loss, gradients and checkpoint.
- Behavior and assumptions: Autocast, GradScaler, gradient clipping, AdamW, cosine scheduler and state recovery.

```python
174:         for step,(x, target, _) in enumerate(bar):
175:             x, target = x.to(device), target.to(device)
176:             with torch.autocast(device_type=device.type,enabled=amp_enabled):
177:                 pred = model(x); raw_loss = loss_fn(pred, target)[0]
178:                 # Normalize a short final accumulation window by its actual
179:                 # number of microbatches instead of underweighting it.
180:                 window_start=(step//accum)*accum
181:                 window_size=min(accum,len(train)-window_start)
182:                 loss=raw_loss/window_size
183:             if not torch.isfinite(raw_loss):
184:                 raise FloatingPointError(f"non-finite training loss at epoch={epoch+1} step={step}")
185:             scaler.scale(loss).backward()
186:             do_step=(step+1)%accum==0 or step+1==len(train)
187:             if do_step:
188:                 scaler.unscale_(opt)
189:                 # FP16 overflow is handled by GradScaler: skip the update and
190:                 # lower the scale. FP32 nonfinite gradients remain fatal.
191:                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp_enabled)
192:                 previous_scale=scaler.get_scale()
193:                 scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
194:                 if scaler.get_scale()<previous_scale:
195:                     print(f"AMP overflow: skipped optimizer update; scale={scaler.get_scale()}")
196:             running.append(raw_loss.item())
197:             bar.set_postfix(loss=f"{loss.item():.4f}")
198:         sched.step()
199:         metrics = validate(model, val, loss_fn, device) if val is not None else None
```

## Evaluation/inference

- Paper concept: Evaluation/inference.
- Implemented symbol: `scripts/test_restoration.py::predict` (AST symbol lines 98-103).
- Input: H x W x 3 RGB float. Output: same-size RGB float.
- Behavior and assumptions: Uses _restore_tensor symmetric multiple-of-128 padding; FP32 full resolution, no tiling.

```python
98: def predict(model, degraded, device, pad_multiple):
99:     x = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0).to(device)
100:     output, elapsed = _restore_tensor(model, x, device, pad_multiple)
101:     if output.shape != x.shape or not torch.isfinite(output).all():
102:         raise RuntimeError("Model returned an invalid shape or nonfinite pixels")
103:     return output, output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), elapsed
```

