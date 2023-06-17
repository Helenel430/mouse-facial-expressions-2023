# -*- coding: utf-8 -*-
import json
import logging
import random
from pathlib import Path

import click
import mlflow

import numpy as np

# import torch
from dotenv import find_dotenv, load_dotenv

# from torch import nn
# from torch.utils.data import DataLoader, Dataset


@click.command(help="Train a Model")
@click.option("--training_data", default="data/processed", type=click.Path())
@click.option("--epochs", type=click.INT, default=10, help="Number of training epochs.")
@click.option("--learning_rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--seed", type=click.INT, default=97531, help="Seed random number generators.")
def main(**kwargs):
    """Train a model."""
    logger = logging.getLogger(__name__)
    logger.info(f"beginning model run {Path(__file__).parts[-1]}")

    training_data = kwargs["training_data"]
    epochs = kwargs["epochs"]
    learning_rate = kwargs["learning_rate"]
    seed = kwargs["seed"]

    # Deterministic Setup
    logger.info(f"Setting random seed {seed}")
    random.seed(seed)
    # np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(seed)
    # # For seeding dataloaders
    # g = torch.Generator() 
    # g.manual_seed(seed)

    with mlflow.start_run() as active_run:
        # log all options or manually specify which ones
        logger.info("logging parameters\n" + json.dumps(kwargs, indent=4))
        mlflow.log_params(kwargs)

        logger.info("loading datasets")
        # TODO load dataset and data loader

        logger.info("beginning model training")
        # TODO implement model training

        logger.info("running test evaluation")
        # TODO report a test loss, required for hparam optimization
        # NOTE it may be advisable to have train/val/test and a separate test holdout
        test_result = 1 / abs(np.log(abs(learning_rate - 0.01)))  # dummy value
        mlflow.log_metric("test_loss", test_result)
        logger.info(f"test result: {test_result:.3f}")

        # signature = mlflow.models.signature.infer_signature(x, x_hat)
        # mlflow.pytorch.log_model(model, "mymodel", signature=signature)

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
