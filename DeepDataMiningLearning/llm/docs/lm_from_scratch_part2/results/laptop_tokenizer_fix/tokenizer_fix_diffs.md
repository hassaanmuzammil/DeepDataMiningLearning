metric `val_bpb` (lower is better); seeds [42, 1337]; noise floor (median within-run seed std) = 0.0055

| run | mean ± std | per seed | non-emb params | tokens | train time | tok/s | peak MB |
|---|---:|---|---:|---:|---:|---:|---:|
| `gpt2_baseline` | 2.8543 ± 0.0202 | 2.8686 / 2.8400 | 3.15M | 16.4M | 1.42 min | 0.19M | 4620 |
| `gpt2_early_stop` | 2.2370 ± 0.0010 | 2.2363 / 2.2377 | 3.15M | 2.0M | 0.18 min | 0.19M | 4620 |
| `gpt2_early_stop_500` | 2.1410 ± 0.0067 | 2.1363 / 2.1457 | 3.15M | 4.1M | 0.36 min | 0.19M | 4620 |
| `gpt2_dropout` | 2.3543 ± 0.0043 | 2.3574 / 2.3513 | 3.15M | 16.4M | 1.42 min | 0.19M | 4620 |
| `gpt2_dropout_wd` | 2.3554 ± 0.0009 | 2.3560 / 2.3548 | 3.15M | 16.4M | 1.42 min | 0.19M | 4620 |
| `gpt2_small` | 3.1734 ± 0.0148 | 3.1838 / 3.1629 | 0.79M | 16.4M | 1.10 min | 0.25M | 4492 |

| comparison (B − A) | mean diff | per seed | paired p | resolved? |
|---|---:|---|---:|---|
| `gpt2_early_stop` − `gpt2_baseline` | -0.6173 | -0.6323 / -0.6022 | 0.015 | yes |
| `gpt2_early_stop_500` − `gpt2_baseline` | -0.7133 | -0.7323 / -0.6942 | 0.017 | yes |
| `gpt2_dropout` − `gpt2_baseline` | -0.5000 | -0.5112 / -0.4887 | 0.014 | yes |
| `gpt2_dropout_wd` − `gpt2_baseline` | -0.4989 | -0.5125 / -0.4852 | 0.017 | yes |
| `gpt2_small` − `gpt2_baseline` | +0.3191 | +0.3152 / +0.3229 | 0.008 | yes |
