import pickle
import pandas as pd
import os
import re
import torch
import sklearn

from torch.utils.data import Dataset
from pathlib import Path
from mouse_facial_expressions.paths import *
from skimage.io import imread
import torchvision
from torchmetrics import Accuracy
from mouse_facial_expressions.data.make_datasets import get_treatment_video_dataframe

class Task1FoldDataset(Dataset):
    def __init__(self, samples, df, transform=None):
        super().__init__()
        self.samples = samples
        self.df = df
        self.transform = transform
        self.frame_dir = Path(get_extracted_frames_folder())

    def get_class_weights(self):
        labels = [item['label'] for item in self.samples]
        weights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', 
            classes=sorted(list(set(labels))),
            y=labels
        )
        return weights
    
    @property
    def indices_in_samples(self):
        return list({i for s in self.samples for i in s['indices']})
    
    @property
    def data_in_samples(self):
        return self.df.loc[self.indices_in_samples]
    
    def get_eval_mask(self):
        """Create a mask which is 
        - `0` for recordings that are present in this dataset
        - `1` for recordings not present in this dataset
        - `-1` for recordings that are missing
        """
        # Get all videos
        df = get_treatment_video_dataframe()
        df = df[['video', 'mouse', 'treatment', 'recording']]
        df = df.drop_duplicates()

        training_videos = self.data_in_samples.video.unique()
        temp = df.copy()
        temp['can_test'] = 0
        temp.loc[~df.video.isin(training_videos), 'can_test'] = 1
        temp = temp.pivot(index=['mouse'], columns=['recording'], values=['can_test']).sort_index()
        temp = temp.fillna(-1).astype(int).droplevel(0, axis=1)
        return temp
    
    def __len__(self):
        return len(self.samples)
    
    def get_image(self, imagepath):
        return self.transform(imread(self.frame_dir / imagepath))
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        indices = sample['indices']
        images = torch.stack(self.df.loc[indices].image.apply(self.get_image).tolist())
        return images, label

class Task1Folds:
    def __init__(self, version='1.0', train_augmentation='TrivialAugmentWide'):
        if train_augmentation == 'TrivialAugmentWide':
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.TrivialAugmentWide(),
                torchvision.transforms.ToTensor()
            ])
        elif train_augmentation is None or train_augmentation.lower() == 'none':
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
        task1_path = get_task_folder(version)
        self.df = pd.read_pickle(task1_path / 'dataset_df.pkl')

        folds = task1_path.glob('fold*.pkl')
        fold_df = pd.DataFrame({'foldpath': folds})
        fold_df['fold_index'] = fold_df.foldpath.apply(lambda x: int(re.match('.*(\d+)', x.parts[-1]).group(1)))
        fold_df = fold_df.sort_values('fold_index')
        fold_df = fold_df.set_index('fold_index')
        self.fold_df = fold_df
        
    def __len__(self):
        return len(self.fold_df)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        
        row = self.fold_df.loc[idx]
        with open(row.foldpath, 'rb') as fp:
            fold_data = pickle.load(fp)
            
        train = fold_data['train']
        train_dataset = Task1FoldDataset(samples=train, df=self.df, transform=self.train_transform)
        
        test = fold_data['test']
        test_dataset = Task1FoldDataset(samples=test, df=self.df, transform=self.test_transform)
        return train_dataset, test_dataset
    

