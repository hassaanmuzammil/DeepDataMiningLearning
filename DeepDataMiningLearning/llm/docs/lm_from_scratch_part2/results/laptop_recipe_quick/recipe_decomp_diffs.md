metric `val_bpb` (lower is better); seeds [42, 1337]; noise floor (median within-run seed std) = 0.0021

| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |
|---|---:|---|---:|---:|---:|---:|---:|
| `textbook` | 2.4890 ± 0.0039 | 2.4862 / 2.4917 | 3.16M | 16.4M | 0.16 min | 1.68M | 588 |
| `tb_rope` | 2.1115 ± 0.0011 | 2.1123 / 2.1108 | 3.16M | 16.4M | 0.18 min | 1.50M | 601 |
| `tb_rmsnorm` | 2.4895 ± 0.0055 | 2.4856 / 2.4934 | 3.16M | 16.4M | 0.18 min | 1.55M | 660 |
| `tb_nobias` | 2.4988 ± 0.0017 | 2.5000 / 2.4976 | 3.15M | 16.4M | 0.14 min | 1.90M | 588 |
| `tb_nope` | 2.5075 ± 0.0008 | 2.5069 / 2.5081 | 3.16M | 16.4M | 0.16 min | 1.74M | 584 |
| `rr_learned` | 2.4993 ± 0.0026 | 2.5011 / 2.4974 | 3.15M | 16.4M | 0.16 min | 1.67M | 659 |
| `rmsnorm_rope` | 2.1241 ± 0.0021 | 2.1256 / 2.1227 | 3.15M | 16.4M | 0.19 min | 1.41M | 672 |

| comparison (B − A) | mean diff | per seed | paired p | resolved? |
|---|---:|---|---:|---|
| `tb_rope` − `textbook` | -0.3774 | -0.3739 / -0.3810 | 0.006 | yes |
| `tb_rmsnorm` − `textbook` | +0.0006 | -0.0005 / +0.0017 | 0.698 | not resolved |
| `tb_nobias` − `textbook` | +0.0099 | +0.0139 / +0.0059 | 0.245 | yes |
| `tb_nope` − `textbook` | +0.0186 | +0.0208 / +0.0164 | 0.076 | yes |
| `rmsnorm_rope` − `rr_learned` | -0.3751 | -0.3755 / -0.3748 | 0.001 | yes |
