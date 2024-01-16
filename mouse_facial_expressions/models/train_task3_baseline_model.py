# -*- coding: utf-8 -*-
import json
import logging
import random
from pathlib import Path
import pandas as pd

import click
import mlflow
import json
import numpy as np
import pickle 

import torchmetrics
import torch
import torchvision
import lightning.pytorch as pl 
import re 

from lightning.pytorch.callbacks import Callback
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader
from mouse_facial_expressions.data.datasets import get_task_folder, Task1FoldDataset

class Task3Folds:
    def __init__(self, sex, version='3.0', train_augmentation='TrivialAugmentWide'):
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

        self.sex = sex
        task1_path = get_task_folder(version)
        self.df = pd.read_pickle(task1_path / 'dataset_df.pkl')

        folds = (task1_path / sex).glob('fold*.pkl')
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
    
    
class DeepSet(pl.LightningModule):
    def __init__(self, config, class_weights):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.model_ = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = self.model_.fc.in_features
        n_classes = 2 
        features = 1
        self.model_.fc = torch.nn.Linear(in_features, features)
        self.fc = torch.nn.Linear(features, n_classes) # The purpose of this layer is really just to add bias
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=config['label_smoothing'])
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)

        
    def configure_optimizers(self):
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        momentum = self.config['momentum']
        optimizer = torch.optim.SGD(params=self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.float()
        return self(x)
    
    def on_train_epoch_start(self):
        self.train_acc.reset()
        self.train_losses = []
        self.learning_rates = []
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        epoch = self.trainer.current_epoch
        mlflow.log_metric("train_acc", self.train_acc.compute(), step=epoch)
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        mlflow.log_metric("learning_rate", learning_rate, step=epoch) 
        mlflow.log_metric("train_loss", np.mean(self.train_losses), step=epoch) 
        self._logger.info(
            f"Train epoch {epoch}, "
            f"loss {np.mean(self.train_losses):.03f}, "
            f"accuracy {self.train_acc.compute():.03f}"
        )
        return super().on_train_epoch_end()
    
    def on_validation_start(self) -> None:
        self.val_acc.reset()
        self.val_losses = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        epoch = self.trainer.current_epoch
        mlflow.log_metric("val_acc", self.val_acc.compute(), step=epoch)
        mlflow.log_metric("val_loss", np.mean(self.val_losses), step=epoch) 
        self._logger.info(
            f"Validation {epoch}, "
            f"loss {np.mean(self.val_losses):.03f}, "
            f"accuracy {self.val_acc.compute():.03f}"
        )
        return super().on_validation_epoch_end()
    
    def training_step(self, batch, batch_idx):
        y_hat = self.predict_step(batch, batch_idx)
        x, y = batch
        y = y.long()
        loss = self.criterion(y_hat, y)
        self.train_losses.append(loss.item())
        self.train_acc(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self.predict_step(batch, batch_idx)
        x, y = batch
        y = y.long()
        loss = self.criterion(y_hat, y)
        self.val_losses.append(loss.item())
        self.val_acc(y_hat, y)
        return loss
    
    def forward(self, x):
        # Flatten (batch, set, images) -> (batch x set, images)
        s = x.shape
        x = x.flatten(0, 1)
        
        # Pass through model
        z = self.model_(x)
        z = z.reshape(*s[:2], z.shape[-1])
        z = z.mean(dim=1) # Mean over each image (dims are batch, image, class preds)
        y_hat = self.fc(z)
        return y_hat


@click.command(help="Train a Model")
# @click.option("--training_data", default="data/processed", type=click.Path())
@click.option("--model", type=click.INT, default=10, help="Number of training epochs.")
@click.option("--epochs", type=click.INT, default=10, help="Number of training epochs.")
@click.option("--learning_rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--train_batch_size", type=click.INT, default=20)
@click.option("--test_batch_size", type=click.INT, default=50)
@click.option("--warmup_steps", type=click.INT, default=0)
@click.option("--momentum", type=click.FLOAT, default=0.9)
@click.option("--weight_decay", type=click.FLOAT, default=1e-4)
@click.option("--warmup_decay", type=click.FLOAT, default=0.0001)
@click.option("--label_smoothing", type=click.FLOAT, default=0.1)
@click.option("--dataset_version", type=click.STRING, default='3.0')
@click.option("--train_augmentation", type=click.STRING, default='TrivialAugmentWide')
@click.option("--seed", type=click.INT, default=97531, help="Seed random number generators.")
def main(**kwargs):
    """Train a model."""
    logger = logging.getLogger(__name__)
    logger.info(f"beginning model run {Path(__file__).parts[-1]}")

    config = kwargs
    epochs = kwargs["epochs"]
    learning_rate = kwargs["learning_rate"]
    seed = kwargs["seed"]
    
    with mlflow.start_run(nested=True) as active_run:
        logger.info("Started mlflow crossvalidation run %s", active_run.info.run_id)
        
        # log all options or manually specify which ones
        logger.info("Logging parameters\n" + json.dumps(kwargs, indent=4))
        mlflow.log_params(kwargs)
        
        for sex in ['male', 'female']:
            logger.info("Processing sex %s", sex)
            cv = Task3Folds(sex, version=config['dataset_version'], train_augmentation=config['train_augmentation'])
            for fold, (train_dataset, test_dataset) in enumerate(cv):
                logger.info("Train size %i", len(train_dataset))
                logger.info("Test size %i", len(test_dataset))

                with mlflow.start_run(nested=True) as child_run:
                    logger.info("Starting training for fold %i with seed %i", fold, seed)
                    
                    logger.info("Logging parameters\n" + json.dumps(kwargs, indent=4))
                    mlflow.set_tag('sex', sex)
                    mlflow.log_params(kwargs)
                    
                    random.seed(seed)
                    np.random.seed(seed)
                    # torch.use_deterministic_algorithms(True)
                    torch.manual_seed(seed)
                    g = torch.Generator()
                    g.manual_seed(seed)
                    
                    logger.info("Creating dataloader")
                    train_dataloader = DataLoader(train_dataset, batch_size=20, num_workers=8, shuffle=True, generator=g)
                    test_dataloader = DataLoader(test_dataset, batch_size=40, num_workers=8)
                    weights = train_dataset.get_class_weights()
                    weights = torch.from_numpy(weights).float()
                    model = DeepSet(config=config, class_weights=weights)
                    
                    loggers = []
                    callbacks = []
                    trainer = pl.Trainer(
                        max_epochs=config['epochs'],
                        callbacks=callbacks,
                        logger=loggers,
                        enable_checkpointing=False,
                        accelerator='gpu', 
                        devices=[0], 
                    )
                    trainer.fit(model, train_dataloader, test_dataloader)
                    
                    # Logging
                    mlflow.set_tag("fold", fold)
                    
                    mask_df = train_dataset.get_eval_mask()
                    mask_df.to_csv('testable_videos_mask.csv')
                    mlflow.log_artifact("testable_videos_mask.csv", "testable_videos_mask")
                    
                    meta_info = {
                        'train_videos': list(train_dataset.data_in_samples.video.unique()),
                        'test_videos': list(test_dataset.data_in_samples.video.unique())
                    }
                    with open('train_meta.json', 'w') as fp:
                        json.dump(meta_info, fp, indent=4)
                    mlflow.log_artifact("train_meta.json", "meta_info")
                    
                    mlflow.pytorch.log_model(model, "model")
                    logger.info("run complete")
    

        logger.info("model run completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
