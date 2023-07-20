# -*- coding: utf-8 -*-
import click
import logging
import cv2
import numpy as np
import pandas as pd

from itertools import combinations
from deeplabcut import create_new_project, analyze_time_lapse_frames
from deeplabcut.utils.auxiliaryfunctions import edit_config, read_config

from PIL import Image
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.group()
def main():
    pass


@main.command()
@click.argument("name", type=str)
@click.argument("scorer", type=str)
@click.argument("working_directory", type=click.Path(exists=True))
@click.option("--placeholder_video", type=str, default=None)
def create_facial_expression_project(
    name, scorer, working_directory, placeholder_video
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating new project %s", __name__)
    bodyparts = [
        'nose',
        'left_eye',
        'left_ear',
        'right_eye',
        'right_ear'
    ]
    if placeholder_video is None:
        placeholder_video = project_dir/'data/raw/placeholder.mp4'
        
    assert placeholder_video.exists(), """
    Placeholder video not found. Run `make download-data` 
    or specify any video with the `--placeholder_video` option
    """
    
    deeplabcut_project_config = create_new_project(
        project=name,
        experimenter=scorer,
        videos=[placeholder_video], # should be the same size as all other videos
        copy_videos=False,
        working_directory=working_directory)

    edit_config(deeplabcut_project_config, {'bodyparts': bodyparts})
    deeplabcut_project = Path(deeplabcut_project_config).parent
    logger.info('DLC project created at %s', str(deeplabcut_project))


@main.command()
@click.argument("deeplabcut_project", type=click.Path(exists=True))
@click.argument("labeled_data_folder", type=click.Path())
@click.argument("video_folder", type=click.Path(exists=True))
@click.argument("frames_per_video", type=int)
def extract_new_frames_from_videos(
    deeplabcut_project, labeled_data_folder, video_folder, frames_per_video
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("extracting new frames to %s", labeled_data_folder)

    deeplabcut_project = Path(deeplabcut_project)
    labeled_data_folder = deeplabcut_project / "labeled-data" / labeled_data_folder
    video_folder = Path(video_folder)

    labeled_data_folder = Path(labeled_data_folder)
    if not labeled_data_folder.exists():
        labeled_data_folder.mkdir(parents=True)

    videos = list(video_folder.glob("*.mp4"))
    logger.info("Found %i videos", len(videos))
    
    shapes = set()
    logger.info("Checking video sizes")
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        shapes.add((width, height))
        logger.info("Video %s dimensions %ix%i", video.parts[-1], width, height)
        
    assert len(shapes) == 1, "Videos cannot be different sizes!"
    
    logger.info("Extracting frames")
    for video in videos:
        logger.info("Extracting frames from video %s", video.parts[-1])
        cap = cv2.VideoCapture(str(video))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames == 0:
            logger.exception("No frames found for video %s", video.parts[-1])
            continue

        selected_frames = np.random.choice(np.arange(frames), frames_per_video)
        fname, _ = Path(video).parts[-1].split(".")

        for i in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = Image.fromarray(frame)
            frame.save(labeled_data_folder / f"{fname}_frame{i:05}.png")
            
        cap.release()
    

@main.command()
@click.argument("deeplabcut_project", type=click.Path(exists=True))
@click.argument("labeled_data_folder", type=click.Path(exists=True))
@click.argument("scorer", type=str)
def automate_labeling(
    deeplabcut_project, labeled_data_folder, scorer
):
    logger = logging.getLogger(__name__)
    logger.info("extracting new frames to %s", labeled_data_folder)

    deeplabcut_project = Path(deeplabcut_project).resolve()
    labeled_data_folder = Path(labeled_data_folder).resolve()
    
    analyze_time_lapse_frames(
        deeplabcut_project / 'config.yaml',
        labeled_data_folder)
    
    dlc_file = next(labeled_data_folder.glob('*DLC*.h5'))
    generated_labels = pd.read_hdf(dlc_file)
    
    images = generated_labels.index
    index = pd.MultiIndex.from_tuples(('labeled-data', labeled_data_folder.parts[-1], l) for l in images)
    columns = pd.MultiIndex.from_tuples(
        [(scorer,b,c) for s,b,c in generated_labels.columns],
        names=['scorer', 'bodyparts', 'coords']
    )
    reshaped_df = generated_labels.copy()
    reshaped_df = reshaped_df.set_index(index)
    reshaped_df.columns = columns

    # Filter out bodyparts that are too close
    counter = 0
    bodyparts = reshaped_df.columns.get_level_values('bodyparts').unique()
    pairs = combinations(bodyparts, r=2)
    threshold = 5
    for idx, row in reshaped_df.iterrows():
        for bp1, bp2 in pairs:
            x1, y1, c1 = row.loc[:, bp1, ['x', 'y', 'likelihood']]
            x2, y2, c2 = row.loc[:, bp2, ['x', 'y', 'likelihood']]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if distance is not np.nan and distance < threshold:
                if c1 > c2:
                    reshaped_df.loc[idx, pd.IndexSlice[:,bp2,:]] = np.nan
                elif c2 > c1:
                    reshaped_df.loc[idx, pd.IndexSlice[:,bp1,:]] = np.nan
                
                counter += 1
                
    # Filter out confidence
    likelihood = reshaped_df.xs('likelihood', level='coords', axis=1)
    confident_labels = reshaped_df.where(likelihood > 0.6)
    images = confident_labels.index
    confident_labels = confident_labels.drop(columns='likelihood', level='coords')
    confident_labels = confident_labels.dropna(axis=0, how='all') # drop rows with no values
    
    confident_labels.to_hdf(labeled_data_folder / f'CollectedData_{scorer}.h5', '/keypoints')
    confident_labels.to_csv(labeled_data_folder / f'CollectedData_{scorer}.csv')

    
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
