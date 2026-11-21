metric `val_bpb` (lower is better); seeds [7, 42, 1337, 2024]; noise floor (median within-run seed std) = 0.0040

| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |
|---|---:|---|---:|---:|---:|---:|---:|
| `gelu` | 2.1241 ± 0.0021 | 2.1256 / 2.1225 | 3.15M | 16.4M | 0.19 min | 1.45M | 672 |
| `swiglu` | 2.0381 ± 0.0034 | 2.0357 / 2.0405 | 3.15M | 16.4M | 0.21 min | 1.28M | 715 |
| `swiglu_lr5e-4` | 2.1109 ± 0.0051 | 2.1073 / 2.1145 | 3.15M | 16.4M | 0.21 min | 1.28M | 715 |
| `swiglu_lr2e-3` | 2.0138 ± 0.0109 | 2.0215 / 2.0061 | 3.15M | 16.4M | 0.21 min | 1.28M | 715 |
| `swiglu_wide` | 2.0218 ± 0.0006 | 2.0214 / 2.0222 | 4.20M | 16.4M | 0.23 min | 1.19M | 818 |

| comparison (B − A) | mean diff | per seed | paired p | resolved? |
|---|---:|---|---:|---|
| `swiglu` − `gelu` | -0.0859 | -0.0898 / -0.0821 | 0.029 | yes |
| `swiglu_lr5e-4` − `gelu` | -0.0132 | -0.0183 / -0.0080 | 0.238 | yes |
| `swiglu_lr2e-3` − `gelu` | -0.1103 | -0.1041 / -0.1164 | 0.036 | yes |
| `swiglu_wide` − `gelu` | -0.1022 | -0.1042 / -0.1003 | 0.012 | yes |
| `swiglu_lr5e-4` − `swiglu` | +0.0728 | +0.0715 / +0.0741 | 0.011 | yes |
| `swiglu_lr2e-3` − `swiglu` | -0.0243 | -0.0143 / -0.0344 | 0.250 | yes |
| `swiglu_wide` − `swiglu` | -0.0163 | -0.0143 / -0.0182 | 0.077 | yes |
