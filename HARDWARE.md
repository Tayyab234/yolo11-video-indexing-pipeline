# Hardware and execution

The 50 Parquet files were processed across three environments. Pipeline development
and experiments on sample data were performed on my local laptop.

| Environment | Parquet files processed | GPU | System RAM | Disk | Operating system | CPU model and cores | Approximate processing time |
|---|---:|---|---|---|---|---|---|
| Rented RunPod instance | 35 | NVIDIA RTX 4090, 24 GB VRAM | 41 GB | 50 GB allocated; SSD/HDD type not recorded | Ubuntu; version not recorded | 6 vCPUs; physical CPU model and core count not recorded | 6–7 hours |
| Kaggle notebook | 10 | Two NVIDIA Tesla T4 GPUs, 16 GB VRAM each; script used GPU 0 only | Default session allocation; exact amount not recorded | 20 GB saved-output allowance in /kaggle/working, plus temporary scratch space; total capacity and SSD/HDD type not recorded | Exact OS/distribution not recorded | Not recorded | 11–12 hours |
| Google Colab notebook | 5 | NVIDIA Tesla T4, 16 GB VRAM | Default session allocation; exact amount not recorded | Default session allocation; capacity and SSD/HDD type not recorded | Exact OS/distribution not recorded | Not recorded | Approximately 5 hours |

The durations above are approximate reported processing times, not exact timings
from a single consolidated run log. Their sum is approximately 22–24 hours of
execution across the three environments. This sum is not a measured wall-clock
duration from the start of the first job to completion of the last job; jobs may
have overlapped. A single overall start-to-finish duration was not recorded.
Whether dataset downloads and environment setup are included in these estimates
was not recorded.

## Local development

- Computer: local Lenovo laptop.
- GPU: NVIDIA Quadro T2000, 4 GB VRAM.
- Operating system: Windows; exact edition/version not recorded.
- Python environment: Conda environment `ds-gpu`.
- CPU model and core count: not recorded.
- System RAM: 16GB RAM
- Disk type: SSD
- Purpose: develop and benchmark the pipeline on sample images and videos. The
  full 50-file dataset was processed in the remote environments listed above.




