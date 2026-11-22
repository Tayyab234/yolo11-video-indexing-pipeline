"""FineVideo indexing with bounded decoding and certified per-second stopping."""
import argparse
import io
import json
import math
import queue
import threading
from collections import Counter
from pathlib import Path

TARGETS = {0: 'person', 2: 'car', 16: 'dog', 63: 'laptop', 67: 'cell phone'}


def floor_median_batch(values):
    if not values:
        return 0
    ordered = sorted(values)
    n = len(ordered)
    return (int(ordered[(n - 1) // 2]) + int(ordered[n // 2])) // 2


def median_fixed_batch(known, total):
    remaining = total - len(known)
    if remaining < 0:
        raise ValueError('Known frames exceed total frames')
    if total == 0:
        return True, 0
    if remaining == 0:
        return True, floor_median_batch(known)
    lower = floor_median_batch(known + [0] * remaining)
    ordered = sorted(known + [float('inf')] * remaining)
    left, right = ordered[(total - 1) // 2], ordered[total // 2]
    if math.isinf(left) or math.isinf(right):
        return False, None
    upper = (int(left) + int(right)) // 2
    return (True, lower) if lower == upper else (False, None)


def objects_fixed_batch(counts, remaining, threshold):
    return all(counts[c] >= threshold or counts[c] + remaining < threshold for c in TARGETS)


def choose_adaptive_batch(counts, people_counts, total_frames, threshold, max_batch=8):
    processed = len(people_counts)
    remaining = total_frames - processed
    if remaining <= 0:
        return 0
    if processed == 0:
        return min(max_batch, remaining)
    unresolved = []
    for cls in TARGETS:
        count = counts[cls]
        if count >= threshold or count + remaining < threshold:
            continue
        unresolved.append(min(threshold - count, count + remaining - threshold + 1))
    distance = max(unresolved) if unresolved else 0
    median_ready, _ = median_fixed_batch(people_counts, total_frames)
    if distance <= 1 and median_ready:
        size = 1
    elif distance <= 2:
        size = 2
    elif distance <= 4:
        size = 3
    else:
        size = max_batch
    return min(size, max_batch, remaining)


def predict_batch(model, frames, device):
    import numpy as np
    if not frames:
        return []
    results = model.predict(source=frames, imgsz=640, conf=0.4, iou=0.7,
                            agnostic_nms=False, device=device, verbose=False)
    if len(results) != len(frames):
        raise RuntimeError('YOLO result count does not match input batch')
    predictions = []
    for result in results:
        classes = result.boxes.cls.int().cpu().numpy()
        detected = {int(c) for c in classes if int(c) in TARGETS}
        predictions.append((detected, int(np.count_nonzero(classes == 0))))
    return predictions


def iter_seconds(data):
    import av
    import cv2
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        origin = None
        current = 0
        frames = []
        seen = False
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Frame has no presentation timestamp')
            timestamp = frame.pts * frame.time_base
            if origin is None:
                origin = timestamp
            second = math.floor(timestamp - origin)
            if second < current:
                raise ValueError('Non-monotonic frame timestamp')
            if second != current:
                yield current, frames
                for gap in range(current + 1, second):
                    yield gap, []
                frames = []
                current = second
            # Do not silently index frames FFmpeg identifies as corrupt.
            if frame.is_corrupt:
                raise ValueError('Corrupt decoded frame')
            image = frame.to_ndarray(format='bgr24')
            # PyAV exposes display-matrix rotation on decoded frames.
            angle = float(getattr(frame, 'rotation', 0) or stream.metadata.get('rotate', 0)) % 360
            if angle:
                if angle == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                else:
                    raise ValueError(f'Unsupported rotation: {angle}')
            frames.append(image)
            seen = True
        if not seen:
            raise ValueError('No video frames decoded')
        yield current, frames


def threaded_seconds(data, size=2):
    fifo = queue.Queue(maxsize=size)
    stop = threading.Event()
    done = object()

    def put(item):
        while not stop.is_set():
            try:
                fifo.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for item in iter_seconds(data):
                if stop.is_set():
                    break
                put(item)
        except Exception as exc:
            put(exc)
        finally:
            put(done)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = fifo.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError('Decoder did not stop')


def process_video(model, data, device, batch_size, full_frames):
    seconds = threaded_seconds(data)
    try:
        for second, frames in seconds:
            counts = Counter()
            people = []
            total = len(frames)
            processed = 0
            final_median = None
            while processed < total:
                if not full_frames and people:
                    median_ready, median_value = median_fixed_batch(people, total)
                    if objects_fixed_batch(counts, total - processed, (total + 1) // 2) and median_ready:
                        final_median = median_value
                        break
                threshold = (total + 1) // 2
                step = choose_adaptive_batch(counts, people, total, threshold, max_batch=batch_size)
                batch = frames[processed:processed + step]
                for detected, count in predict_batch(model, batch, device):
                    counts.update(detected)
                    people.append(count)
                processed += len(batch)
            median = final_median if final_median is not None else floor_median_batch(people)
            yield {'second': second,
                   'objects': [name for c, name in TARGETS.items() if total and 2 * counts[c] >= total],
                   'people': median}
    finally:
        seconds.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--weights",type=Path,default=Path("/workspace/yolo11l.pt"),)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--full-frames', action='store_true', help='Disable certified early stopping')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    files = sorted(args.input.glob('*.parquet'))
    if not files:
        parser.error('No .parquet files directly inside input folder')
    if not args.weights.is_file():
        parser.error('Weights missing: run prepare.py before the offline run')
    import pyarrow.parquet as pq
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', buffering=1024 * 1024) as output:
        def write(record):
            output.write(json.dumps(record, separators=(',', ':')) + '\n')

        for path in files:
            try:
                parquet = pq.ParquetFile(path)
            except Exception as exc:
                # No row IDs can be recovered from an unreadable file.
                write({'video': f'{path.name}:0', 'error': f'Cannot read Parquet: {exc}'[:300]})
                output.flush()
                continue
            row = 0
            try:
                for group in range(parquet.num_row_groups):
                    group_rows = parquet.metadata.row_group(group).num_rows
                    try:
                        table = parquet.read_row_group(group, columns=['mp4'])
                    except Exception as exc:
                        for offset in range(group_rows):
                            write({'video': f'{path.name}:{row + offset}', 'error': str(exc)[:300]})
                        row += group_rows
                        continue
                    for value in table.column('mp4'):
                        video_id = f'{path.name}:{row}'
                        row += 1
                        try:
                            data = value.as_py()
                            if isinstance(data, dict):
                                data = data.get('bytes')
                            if not isinstance(data, (bytes, memoryview)) or not data:
                                raise ValueError('Missing MP4 bytes')
                            for record in process_video(model, data, args.device, args.batch_size, args.full_frames):
                                write({'video': video_id, **record})
                        except Exception as exc:
                            write({'video': video_id, 'error': str(exc)[:300]})
                            print(f'{video_id}: {exc}', flush=True)
                        output.flush()
                        print(f'Finished {video_id}', flush=True)
                    del table
            finally:
                parquet.close()



if __name__ == "__main__":
    
    main()
    
