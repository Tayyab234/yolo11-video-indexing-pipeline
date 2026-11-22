"""Download official weights before the timed, offline run."""
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).resolve().parent)
    YOLO('yolo11l.pt')
    print('Official weights ready: yolo11l.pt')
