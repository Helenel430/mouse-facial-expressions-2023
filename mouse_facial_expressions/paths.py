import os
from pathlib import Path

def get_raw_video_csv():
    f = Path(os.environ["MFE_RAW_CSV_FOLDER"])
    f = f / f'raw_videos_{os.environ.get("MFE_VERSION")}.csv'
    return str(f)

def get_raw_treatments_csv():
    f = Path(os.environ["MFE_RAW_CSV_FOLDER"])
    f = f / f'treatments_{os.environ.get("MFE_VERSION")}.csv'
    return str(f)

def get_raw_video_folder():
    try:
        f = Path(os.environ["MFE_RAW_VIDEO_FOLDER"])
        return str(f)
    except:
        return None


def get_processed_video_folder():
    try:
        f = Path(os.environ["MFE_PROCESSED_VIDEO_FOLDER"])
        f = f / os.environ.get("MFE_VERSION")
        return str(f)
    except:
        return None


def get_dlc_facial_labels_folder():
    try:
        f = Path(os.environ["MFE_DLC_FACIAL_LABELS_FOLDER"])
        f = f / os.environ.get("MFE_VERSION")
        return str(f)
    except:
        return None


def get_dlc_facial_project_folder():
    try:
        return os.environ["MFE_DLC_FACIAL_PROJECT_PATH"]
    except:
        return None


def get_extracted_frames_folder():
    try:
        f = Path(os.environ["MFE_EXTRACTED_FRAMES_FOLDER"])
        f = f / os.environ.get("MFE_VERSION")
        return str(f)
    except:
        return None

def get_task_folder(version):
    try:
        f = Path(os.environ['MFE_TASKS']) / f"task-{version}"
        return f
    except:
        return None