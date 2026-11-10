# Published results and protocol context

Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324.

**Contextual comparison only — different datasets and evaluation protocols.**

Paper weather scores verified visually from Table XI final row (PDF page 10): rain 30.25/.949, raindrop 28.26/.911, haze 32.33/.989, snow 31.75/.954. These are averages of involved benchmark datasets, not local per-image group means. Separate single-weather model scores in Tables VI-IX were not substituted.

Table XIII reports 58.34M parameters, 5.746G FLOPs and .086 seconds/image. Section V-B specifies 256x256 efficiency inputs; Section IV-B specifies RTX3090. Local counts are 612,435; local FLOPs are unavailable; .380816 s/image uses variable full-size tests on RTX2000 Ada. No speed ratio or FLOPs advantage is supported.

Paper training: AIR40K, reported 40,561 pairs; 256 crops; AdamW .9/.999, weight decay 1e-4, initial LR 2e-4, cosine minimum 1e-7; flips/90-degree rotations/MixUp .5. Main paper points to supplementary material for additional details; batch size and epochs are unresolved. The individual Table I train counts have a historical five-pair discrepancy (sum 40,556 versus reported 40,561) retained in dataset audit records. Do not invent samples to reconcile it.

Local training: custom 58,766 train / 14,695 test, 128 crops, 20 epochs, batch 8, cosine minimum 1e-6. No official AIR40K membership or matched baseline/ablation. See architecture comparison for the twenty component findings.

Paper image excerpts in this directory are byte-for-byte copies of existing extracted figures. Attribution: Tao Gao, Yuanbo Wen, Kaihao Zhang, Jing Zhang, Ting Chen, Lidong Liu, and Wenhan Luo. Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 34(3), March 2024. DOI: 10.1109/TCSVT.2023.3299324. Full source PDF hash and crop coordinates are in `sources/paper_figure_provenance.json`. The supplied PDF carries a licensed-use notice. These excerpts are supplied for the requested internal research review with attribution. Permission for public redistribution is not established; do not describe them as openly licensed. The full PDF is not placed in the ZIP.
