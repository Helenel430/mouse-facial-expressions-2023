"""Hyperparameter optimization example with optuna
Optuna docs: https://optuna.readthedocs.io/en/stable/
"""
import click
import logging
import mlflow
import optuna
import numpy as np
from functools import partial


class Objective:
    """Objective to optimize with optuna"""

    def __init__(self, tracking_client, experiment_id, seed, metric="test_loss"):
        self.tracking_client = tracking_client
        self.experiment_id = experiment_id
        self.seed = seed
        self.metric = metric

    def __call__(self, trial):
        # TODO Define parameters and ranges to optimize
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        # Create a training run with the hyperparameters
        with mlflow.start_run(nested=True) as child_run:
            # Define which model and what parameters are being run
            p = mlflow.projects.run(
                run_id=child_run.info.run_id,
                uri=".",
                entry_point="train",
                parameters={"learning_rate": str(learning_rate), "seed": str(self.seed)},
                experiment_id=self.experiment_id,
                synchronous=False,
            )

            # Wait for the model to complete
            succeeded = p.wait()

        # Return the run test loss
        if succeeded:
            training_run = self.tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics
            score = metrics[self.metric]
        else:
            self.tracking_client.set_terminated(p.run_id, "FAILED")
            score = np.finfo(np.float64).max

        return score


@click.command(help="Hparam runing with optuna")
@click.option("--n_trials", type=click.INT, default=10, help="Number of optimization runs.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
def main(n_trials, seed):
    logger = logging.getLogger(__name__)

    # Create MLFlow client
    np.random.seed(seed)
    tracking_client = mlflow.MlflowClient()

    # Run the hyper-parameter optimization
    with mlflow.start_run() as run:
        # log parameters here
        mlflow.log_param("seed", seed)
        experiment_id = run.info.experiment_id

        # Create optuna optimizer
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        objective = Objective(
            tracking_client=tracking_client, experiment_id=experiment_id, seed=seed
        )
        study.optimize(objective, n_trials=n_trials)
        logger.info("optimization completed")

    # Find and report the best run
    logger.info("finding best run")
    runs = tracking_client.search_runs(
        [experiment_id], "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
    )
    best_test_score = np.finfo(np.float64).max
    best_run = None
    for r in runs:
        if r.data.metrics[objective.metric] < best_test_score:
            best_run = r
            best_test_score = r.data.metrics[objective.metric]

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metrics({f"best_{objective.metric}": best_test_score})
    logger.info(f"best_{objective.metric}={best_test_score}")
    logger.info(f"best run: {best_run.info.run_id}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
