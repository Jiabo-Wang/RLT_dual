# pi0.5 SFT run `biy4hbcy` — metric summary

Table view of the same numbers the PNGs plot, so nothing depends on color.

## Training (150 logging windows, every 200 steps)

| metric | @200 | @10k | @20k | @30k | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| loss | 0.1837 | 0.0280 | 0.0216 | 0.0189 | 0.0079 | 0.1837 |
| grad norm | 2.234 | 0.299 | 0.321 | 0.321 | 0.291 | 2.234 |
| learning rate | 2.53e-06 | 1.95e-05 | 8.23e-06 | 2.50e-06 | 2.50e-06 | 2.49e-05 |
| update s/step | 6.121 | 6.221 | 6.110 | 6.193 | 6.075 | 6.237 |
| dataloading s/step | 0.0339 | 0.0209 | 0.0204 | 0.0204 | 0.0200 | 0.0339 |
| epochs | 0.01 | 0.72 | 1.45 | 2.17 | 0.01 | 2.17 |

## Hardware (24,650 samples over 51.3 h)

| metric | mean | min | max |
| --- | --- | --- | --- |
| GPU 0 utilization (%) | 99.3 | 0.0 | 100.0 |
| GPU 0 VRAM (%) | 86.7 | 1.3 | 88.1 |
| GPU 0 temp (°C) | 88.7 | 58.0 | 90.0 |
| GPU 0 power (W) | 287.7 | 21.7 | 300.1 |
| GPU 0 SM clock (MHz) | 1537 | 210 | 1860 |
| GPU 1 utilization (%) | 63.7 | 0.0 | 100.0 |
| GPU 1 VRAM (%) | 54.3 | 0.0 | 98.6 |
| host CPU (%) | 2.12 | 2.11 | 6.08 |
| host RAM (%) | 7.40 | 3.74 | 13.10 |
| trainer RSS (MB) | 6471 | 6264 | 33067 |
