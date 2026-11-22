# FineVideo YOLO11l indexer

Python 3.11 and an NVIDIA CUDA-capable GPU are required for the default device.
Package versions mirror the development ds-gpu environment; validate availability
and installation on a fresh Ubuntu machine before submission.

## Ubuntu 22.04 installation

```sh
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.11 is supplied by the grading VM. Run these commands from the cloned
repository directory. If the evaluator already created and activated a Python
virtual environment, skip creating and activating .venv. PyAV wheels include
FFmpeg libraries; no separate FFmpeg executable or Docker is used.

## Preparation (internet required, outside timed execution)

```sh
python prepare.py
```

The evaluator must run this command once after installing requirements, while
internet access is available and before the timed run. prepare.py calls
Ultralytics YOLO("yolo11l.pt") to download the official weights, reusing them if
already present. It sets its working directory to the repository script directory,
so yolo11l.pt is saved beside run.py and prepare.py.

run.py resolves the default weights path using
Path(__file__).with_name("yolo11l.pt"). It does not use a developer-specific
Windows path or depend on the shell's current directory to locate the weights.
Keep the downloaded file in place for the offline run. run.py checks it exists
before loading and exits with a preparation message if it is missing; it does not
download weights during timed execution. Model weights are data, not a Python
dependency; Ultralytics is the YOLO implementation listed in requirements.txt.

Expected layout after preparation:

```text
repository/
  run.py
  prepare.py
  requirements.txt
  README.md
  yolo11l.pt
```

## Timed run (no internet required)

```sh
python run.py --input finevideo/data --output results.jsonl
```

The evaluator runs this command after preparation, with internet disabled, using
their test-data directory in place of finevideo/data. Installation and preparation
are outside the timed run; execution of run.py is timed from start to finish.

Use the directory directly containing the Parquet files. All files in that
directory are processed into ONE output file. Existing output is overwritten.
For the full submission, place all first 50 files in the input directory.

On a small GPU, reduce batch size if necessary:

```sh
python run.py --input finevideo/data --output results.jsonl --batch-size 1
```

Default inference is FP32, batch limit 5, size 640, confidence 0.4, class-aware
NMS IoU 0.7, max detections 300. A bounded CPU decoder overlaps decoding with GPU
work. Frame timestamps define seconds relative to the first decoded frame.
Display-matrix rotation is applied before inference. Missing seconds produce
empty records. Early stopping occurs only when every target object's presence
and the floored median people count are fixed for all remaining nonnegative
people counts. This avoids approximate sampling. The batching and aggregation
helpers follow the supplied notebook implementations, with the command-line batch
limit also enforced for limits below 5. Target classes are filtered after inference.

For an all-frame comparison run:

```sh
python run.py --input finevideo/data --output reference.jsonl --full-frames
```

Compare parsed records by (video, second), not line order. Batch-dependent
numerical differences can still affect detections near confidence/NMS thresholds;
validate on representative videos before the full run.

## Damaged input

Completed seconds remain in the output if a video subsequently fails. The script
adds an error record to the SAME output and continues to the next video. A
partially decoded second is discarded on a decode error; no missing detections
are invented. Recovery within a damaged video is not attempted because dropped
frames can change the reference aggregates. Errors reported by the decoder and
frames marked corrupt are caught; silent codec concealment cannot always be
identified. Readable row-group metadata allows one error record per affected row.
If the entire Parquet header is unreadable, the number of videos is unknowable;
one file-level diagnostic uses the identifier filename:0.

## Submission compression

```sh
python -c "import gzip,shutil; src=open('results.jsonl','rb'); dst=gzip.open('results.jsonl.gz','wb'); shutil.copyfileobj(src,dst); dst.close(); src.close()"
```

Gzip compresses the single results file; it is not a ZIP archive and does not
split results by shard. Retain the original until submission is verified.

The remaining submission files (HARDWARE.md, WRITEUP.md, ai_chats/) must also be
provided. Record actual hardware and measured full-run timings. The brief asks
you to write WRITEUP.md yourself without AI assistance.
