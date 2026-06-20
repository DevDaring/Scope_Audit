# MIRAGE Validity Gap Leaderboard

Native pass rate = correct on slot-(a) only (source benchmark surface test).
MIRAGE-Full = behavioural + CDVA validity instrument pass.
Validity gap = native − MIRAGE-Full (hidden invalidity the source benchmark misses).

## Per benchmark (macro-averaged over models)

| Benchmark | N seeds | Native pass | MIRAGE-Full pass | Validity gap |
|---|---:|---:|---:|---:|
| BBQ | 254 | 72.4% | 18.5% | **39.4%** |
| CrowS-Pairs | 181 | 33.6% | 8.6% | **20.4%** |
| StereoSet | 161 | 26.6% | 3.7% | **17.2%** |

## Per model × benchmark

| Model | Benchmark | Native | MIRAGE-Full | Gap |
|---|---|---:|---:|---:|
| amazon-nova-2-lite | BBQ | 76.4% | nan% | nan% |
| amazon-nova-2-lite | CrowS | 54.7% | nan% | nan% |
| amazon-nova-2-lite | StereoSet | 28.0% | nan% | nan% |
| gemini-2.5-flash | BBQ | 94.9% | nan% | nan% |
| gemini-2.5-flash | CrowS | 22.7% | nan% | nan% |
| gemini-2.5-flash | StereoSet | 35.4% | nan% | nan% |
| gemma-2-2b-it | BBQ | 64.2% | 17.7% | 46.5% |
| gemma-2-2b-it | CrowS | 30.9% | 3.9% | 27.1% |
| gemma-2-2b-it | StereoSet | 20.5% | 3.7% | 16.8% |
| llama-3.1-8b-instruct | BBQ | 73.6% | 31.5% | 42.1% |
| llama-3.1-8b-instruct | CrowS | 67.4% | 29.8% | 37.6% |
| llama-3.1-8b-instruct | StereoSet | 32.3% | 6.8% | 25.5% |
| mistral-medium | BBQ | 85.8% | nan% | nan% |
| mistral-medium | CrowS | 51.9% | nan% | nan% |
| mistral-medium | StereoSet | 36.6% | nan% | nan% |
| phi-4-mini-instruct | BBQ | 17.7% | 0.0% | 17.7% |
| phi-4-mini-instruct | CrowS | 5.5% | 0.0% | 5.5% |
| phi-4-mini-instruct | StereoSet | 3.1% | 0.0% | 3.1% |
| qwen2.5-7b-instruct | BBQ | 76.0% | 24.8% | 51.2% |
| qwen2.5-7b-instruct | CrowS | 12.2% | 0.6% | 11.6% |
| qwen2.5-7b-instruct | StereoSet | 28.0% | 4.3% | 23.6% |
| qwen3-next-80b-a3b | BBQ | 90.6% | nan% | nan% |
| qwen3-next-80b-a3b | CrowS | 23.2% | nan% | nan% |
| qwen3-next-80b-a3b | StereoSet | 28.6% | nan% | nan% |
