"""
1. Rename and crop videos
2. Extract deeplabcut labels
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import os

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from functools import partial


def to_seconds(time_str):
    d = datetime.strptime(time_str, "%M:%S")
    d = timedelta(minutes=d.minute, seconds=d.second)
    return int(d.total_seconds())


def seconds_to_str(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{minutes:02}:{seconds:02}"


def get_original_filename(row):
    return f""


def find_video_from_details(row, video_directory):
    s = f"{row.camera}__*__{row.year:04}{row.month:02}{row.day:02}_{row.hour:02}{row.minutes:02}{row.seconds:02}*.mp4"
    try:
        return next(video_directory.glob(s)).parts[-1]
    except:
        return ""


def preprocess_video(row, input_directory, output_directory):
    if row.discard == 1:
        return

    cmd = "ffmpeg -hide_banner -loglevel error -y"

    # Add a start flag if start time is specified
    if row.start != "0" and row.start != "nan":
        cmd += f" -ss {row.start}"

    # Add a duration flag if end time is specified
    if row.end != "-1" and row.end != "nan":
        if row.start != "0" and row.start != "nan":
            duration = seconds_to_str(to_seconds(row.end) - to_seconds(row.start))
        else:
            duration = row.end

        cmd += f" -t {duration}"

    # Add the video in
    cmd += f' -i "{str(input_directory / row.original_video)}"'

    # Specify codec
    cmd += " -c copy"

    # Output file
    recording_labels = dict(
        enumerate(
            [
                "acclimation",
                "preinjection",
                "1h-postinjection",
                "2h-postinjection",
                "4h-postinjection",
            ]
        )
    )
    fname = f"{row.animal}_{recording_labels[int(row.recording)]}.mp4"
    cmd += f' "{output_directory / fname}"'

    return cmd

@click.group()
def main():
    pass

@main.command()
def list_videos():
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Setup file paths
    project_dir = Path(__file__).resolve().parents[2]
    meta_df = pd.read_csv(project_dir / 'data/raw/raw_videos.csv')
    meta_df["recording"] = meta_df["recording"].astype(str)
    for group, group_df in meta_df.groupby('animal'):
        print(f"{group}: " + ', '.join(group_df.recording.tolist()))

@main.command()
@click.option("--output_folder", default=None)
@click.option("-m", "--mouse", multiple=True, default=None, help='e.g. `m1.0, m2.2, f16.2`')
def rename(output_folder, mouse):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Setup file paths
    project_dir = Path(__file__).resolve().parents[2]
    input_video_folder = Path(os.environ['MFE_RAW_VIDEOS'])
    meta_df = pd.read_csv(project_dir / 'data/raw/raw_videos.csv')
    if output_folder is None:
        output_folder = Path(os.environ['MFE_PROCESSED_VIDEOS'])
    else:
        output_folder = Path(output_folder)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)

    # Match the meta info file to the raw video filenames
    logger.info("Finding video names from the meta information file")
    meta_df["original_video"] = meta_df.apply(
        lambda x: find_video_from_details(x, video_directory=input_video_folder), axis=1
    )

    # Clean up datatypes
    meta_df["start"] = meta_df["start"].astype(str)
    meta_df["end"] = meta_df["end"].astype(str)
    meta_df["recording"] = meta_df["recording"]

    if not output_folder.exists():
        logger.info(f"Output directory not found, creating directory '{output_folder}'")
        output_folder.mkdir(parents=True)

    # if specific mice were specified, print them out 
    if mouse is not None:
        rows = []
        for m in mouse:
            try:
                id, rec = m.split('.')
                rec = int(rec)
                row = meta_df[(meta_df.animal == id) & (meta_df.recording == rec) & (meta_df.discard != 1)].iloc[0]
                rows.append(row)
            except:
                logger.warn(f"Mouse {id} and recording {rec} are not in dataset")

        meta_df = pd.DataFrame(rows)

    # process videos
    for idx, row in meta_df.iterrows():
        cmd = preprocess_video(row, input_video_folder, output_folder)
        if cmd:
            logger.info(f"Processing video with command: `{cmd}`")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
        else:
            logger.info(f"Skipping video: `{row.original_video}`")

    logger.info("Processing data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
