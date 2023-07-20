import numpy as np
import pandas as pd
import re
import logging
import click
import os
import sklearn
import random 
import pickle 
import textwrap

from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from mouse_facial_expressions.paths import *

project_dir = Path(__file__).resolve().parents[2]
load_dotenv(find_dotenv())

@click.group()
def main():
    pass


def get_treatment_video_dataframe():
    treatments_df = pd.read_csv(get_raw_treatments_csv())
    frames_dir = Path(get_extracted_frames_folder())
    
    frames_folder_df = pd.DataFrame(dict(image=list(frames_dir.glob('*/*.png'))))
    frames_folder_df['video'] = frames_folder_df.image.apply(lambda x: x.parts[-2])
    frames_folder_df['mouse'] = frames_folder_df.video.apply(lambda x: re.match('([mf]\d+)', x).group(1))
    frames_folder_df['recording'] = frames_folder_df.video.apply(lambda x: int(re.match('.*rec(\d+)', x).group(1)))
    frames_folder_df['image'] = frames_folder_df.image.apply(lambda x: Path(*x.parts[-2:]))

    raw_videos_df = pd.read_csv(get_raw_video_csv())
    raw_videos_df.recording = raw_videos_df.recording.fillna(-1).astype(int)
    raw_videos_df['video_time'] = raw_videos_df.apply(lambda x: f"{x.hour:02}:{x.minutes:02}", axis=1)
    raw_videos_df['mouse'] = raw_videos_df.animal

    combined_df = treatments_df.merge(frames_folder_df, how='left', on='mouse')
    combined_df = combined_df.merge(raw_videos_df, how='left', on=['mouse', 'recording'])
    combined_df = combined_df.drop_duplicates('image') # Not sure why duplicate frames are coming up
    
    return combined_df

@main.command()
@click.option("--frameset_size", default=5, type=int)
@click.option("--train_size", default=10000, type=int)
@click.option("--test_size", default=1000, type=int)
@click.option("--kfold_splits", default=8, type=int)
@click.option("--seed", default=13641, type=int)
def task1(frameset_size, train_size, test_size, kfold_splits, seed):
    logger = logging.getLogger(__name__)

    output_path = Path(os.environ['MFE_TASKS']) / 'task1'
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    task_description = f"""
    Task1 classification task using sets of {frameset_size} frames

    Classes:
    - 0: All animals at preinjection
    - 1: LPS High dose at 4 hours
    
    To balance the training datset (n={train_size}), classes are first sampled
    randomly then videos are randomly sampled and finally frames.
    
    The testing dataset (n={test_size}) only samples by video, preserving imbalances
    in the dataset.

    Split over {kfold_splits} stratified kfolds grouped by mouse.
    """
    task_description = textwrap.dedent(task_description)
    logger.info(task_description)
    with open(output_path / 'README.txt', 'w') as fp:
        fp.write(task_description)

    logger.info('Seeding %i', seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info('Loading treatment csv')
    combined_df = get_treatment_video_dataframe(
        project_dir / 'data/raw/treatments_20230627.csv', 
        project_dir / 'data/raw/raw_videos_20230627.csv')
    
    logger.info('Removing control mouse (m18) which was identified as having pain/sickness symptoms before experiment start')
    combined_df = combined_df[combined_df.mouse != 'm18'] 

    logger.info('Assigning labels')
    # Label everything a 1
    # Label control situations
    combined_df['label'] = np.nan
    combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'saline'), 'label'] = 0
    combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'low'), 'label'] = 0
    combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'mid'), 'label'] = 0
    combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'high'), 'label'] = 0
    combined_df.loc[(combined_df.recording == 4) & (combined_df.treatment == 'high'), 'label'] = 1
    combined_df = combined_df.dropna(subset='label')
    
    logger.info('Final size of task dataset %i', len(combined_df))
    combined_df_by_class = combined_df.groupby('label')
    for label, group_df in combined_df_by_class:
        logger.info('Class %i has %i frames', label, len(group_df))
    
    logger.info('Saving the dataset')
    combined_df.to_pickle(output_path / 'dataset_df.pkl')

    logger.info('Creating %i stratified kfold splits, grouped by mouse', kfold_splits)
    cv = StratifiedGroupKFold(kfold_splits)
    splits = list(cv.split(combined_df.index, groups=combined_df.mouse, y=combined_df.label))
    for fold, split in enumerate(splits): 
        train, test = split
        train_df = combined_df.loc[combined_df.index[train]]
        test_df = combined_df.loc[combined_df.index[test]]
        
        train_mice = train_df.mouse.unique()
        test_mice = test_df.mouse.unique()
        logger.info('\nSplit %i\n - train mice: %s\n - test mice: %s', 
                    fold, 
                    ', '.join(train_mice),
                    ', '.join(test_mice))
        
        logger.info('Fetching train samples')
        train_label0_videos = train_df[train_df.label==0].video.unique()
        train_label1_videos = train_df[train_df.label==1].video.unique()
        train_samples = []
        for _ in tqdm(list(range(train_size)), leave=False):
            label = random.choice([0,1])
            if label == 1:
                video = random.choice(train_label1_videos)
            else:
                video = random.choice(train_label0_videos)
                    
            video_df = train_df[train_df.video==video]
            indices = np.random.choice(video_df.index, size=frameset_size)
            train_samples.append(dict(indices=indices, label=label))
            
        logger.info('Fetching test samples')
        videos = test_df.video.unique()
        test_samples = []
        for _ in tqdm(list(range(test_size)), leave=False):
            video = random.choice(videos)
            video_df = test_df[test_df.video==video]
            indices = np.random.choice(video_df.index, size=frameset_size)
            label = video_df.iloc[0].label
            test_samples.append(dict(indices=indices, label=label))
        
        logger.info('Saving samples')
        data = dict(train=train_samples, test=test_samples)
        with open(output_path / f'fold{fold}.pkl', 'wb') as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
