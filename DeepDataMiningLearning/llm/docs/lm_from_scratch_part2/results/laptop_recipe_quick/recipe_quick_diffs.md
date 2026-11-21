metric `val_bpb` (lower is better); seeds [42, 1337]; noise floor (median within-run seed std) = 0.0022

| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |
|---|---:|---|---:|---:|---:|---:|---:|
| `textbook` | 2.4890 ± 0.0039 | 2.4862 / 2.4917 | 3.16M | 16.4M | 0.16 min | 1.68M | 588 |
| `rmsnorm_rope` | 2.1241 ± 0.0021 | 2.1256 / 2.1227 | 3.15M | 16.4M | 0.19 min | 1.41M | 672 |
| `wsd_adamw` | 2.0673 ± 0.0004 | 2.0676 / 2.0670 | 3.15M | 16.4M | 0.20 min | 1.39M | 672 |
| `muon_wsd` | 1.8261 ± 0.0022 | 1.8245 / 1.8276 | 3.15M | 16.4M | 0.26 min | 1.06M | 660 |
| `muon_cosine` | 1.8395 ± 0.0058 | 1.8354 / 1.8435 | 3.15M | 16.4M | 0.26 min | 1.05M | 660 |

| comparison (B − A) | mean diff | per seed | paired p | resolved? |
|---|---:|---|---:|---|
| `rmsnorm_rope` − `textbook` | -0.3648 | -0.3606 / -0.3691 | 0.007 | yes |
| `wsd_adamw` − `rmsnorm_rope` | -0.0568 | -0.0580 / -0.0556 | 0.013 | yes |
| `muon_wsd` − `wsd_adamw` | -0.2412 | -0.2430 / -0.2394 | 0.005 | yes |
| `muon_cosine` − `rmsnorm_rope` | -0.2847 | -0.2902 / -0.2791 | 0.012 | yes |
| `muon_wsd` − `muon_cosine` | -0.0134 | -0.0109 / -0.0159 | 0.118 | yes |
| `muon_wsd` − `textbook` | -0.6629 | -0.6616 / -0.6641 | 0.001 | yes |

2x2 factorial: cells rmsnorm_rope | wsd_adamw | muon_cosine | muon_wsd

| effect | mean | per seed | paired p |
|---|---:|---|---:|
| main effect of WSD | -0.0351 | -0.0344 / -0.0358 | 0.012 |
| main effect of Muon | -0.2629 | -0.2666 / -0.2593 | 0.009 |
| interaction (WSD effect with Muon − without) | +0.0435 | +0.0472 / +0.0398 | 0.054 |
