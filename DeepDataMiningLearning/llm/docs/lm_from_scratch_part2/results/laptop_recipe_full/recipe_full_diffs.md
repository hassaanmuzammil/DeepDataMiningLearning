metric `val_bpb` (lower is better); seeds [7, 42, 1337, 2024]; noise floor (median within-run seed std) = 0.0029

| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |
|---|---:|---|---:|---:|---:|---:|---:|
| `textbook` | 1.7871 ± 0.0029 | 1.7908 / 1.7844 / 1.7878 / 1.7853 | 10.65M | 49.2M | 1.25 min | 0.67M | 1128 |
| `rmsnorm_rope` | 1.7497 ± 0.0215 | 1.7439 / 1.7669 / 1.7219 / 1.7664 | 10.62M | 49.2M | 1.45 min | 0.57M | 1314 |
| `wsd_adamw` | 1.6969 ± 0.0097 | 1.6923 / 1.7057 / 1.6854 / 1.7041 | 10.62M | 49.2M | 1.49 min | 0.56M | 1314 |
| `muon_wsd` | 1.6125 ± 0.0023 | 1.6101 / 1.6153 / 1.6113 / 1.6132 | 10.62M | 49.2M | 1.82 min | 0.46M | 1274 |
| `muon_cosine` | 1.6110 ± 0.0015 | 1.6089 / 1.6123 / 1.6111 / 1.6119 | 10.62M | 49.2M | 1.91 min | 0.44M | 1274 |

| comparison (B − A) | mean diff | per seed | paired p | resolved? |
|---|---:|---|---:|---|
| `rmsnorm_rope` − `textbook` | -0.0373 | -0.0470 / -0.0175 / -0.0659 / -0.0189 | 0.050 | yes |
| `wsd_adamw` − `rmsnorm_rope` | -0.0529 | -0.0516 / -0.0612 / -0.0364 / -0.0623 | 0.003 | yes |
| `muon_wsd` − `wsd_adamw` | -0.0844 | -0.0821 / -0.0904 / -0.0741 / -0.0909 | 0.000 | yes |
| `muon_cosine` − `rmsnorm_rope` | -0.1387 | -0.1349 / -0.1546 / -0.1108 / -0.1545 | 0.001 | yes |
| `muon_wsd` − `muon_cosine` | +0.0014 | +0.0012 / +0.0030 / +0.0002 / +0.0013 | 0.091 | not resolved |
| `muon_wsd` − `textbook` | -0.1746 | -0.1807 / -0.1690 / -0.1765 / -0.1721 | 0.000 | yes |

2x2 factorial: cells rmsnorm_rope | wsd_adamw | muon_cosine | muon_wsd

| effect | mean | per seed | paired p |
|---|---:|---|---:|
| main effect of WSD | -0.0257 | -0.0252 / -0.0291 / -0.0181 / -0.0305 | 0.003 |
| main effect of Muon | -0.1115 | -0.1085 / -0.1225 / -0.0925 / -0.1227 | 0.001 |
| interaction (WSD effect with Muon − without) | +0.0543 | +0.0528 / +0.0642 / +0.0366 / +0.0636 | 0.004 |
