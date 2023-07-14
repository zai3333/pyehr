import os
from pathlib import Path

import pandas as pd
import lightning as L

import models
from datasets.loader.unpad import unpad_batch
from metrics import check_metric_is_better, get_all_metrics


class MlPipeline(L.LightningModule):
    """A pipeline for training and testing machine learning models.

    Attributes:
        task (str): Task type, such as "outcome" or "los".
        los_info (dict): Length of Stay (LOS) information.
        model_name (str): Name of the model.
        main_metric (str): Main evaluation metric.
        cur_best_performance (dict): Dictionary to store the current best performance metrics.
        model (torch.nn.Module): Model instance.
        test_performance (dict): Dictionary to store the test performance metrics.
        test_outputs (dict): Dictionary to store the test outputs.
        checkpoint_path (str): Path to save the best model checkpoint.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.task = config["task"]
        self.los_info = config["los_info"]
        self.model_name = config["model"]
        self.main_metric = config["main_metric"]
        self.cur_best_performance = {}

        model_class = getattr(models, self.model_name)
        self.model = model_class(**config)

        self.test_performance = {}
        self.test_outputs = {}
        checkpoint_folder = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/'
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_folder, 'best.ckpt')

    def forward(self, x):
        pass
    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Batch of data.It is large enough to contain the whole training set.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        self.model.fit(x, y) # y contains both [outcome, los]
    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: Batch of data.
            batch_idx (int): Index of the current batch.

        Returns:
            main_score: Main evaluation score.
        """
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        y_hat = self.model.predict(x) # y_hat is the prediction results, outcome or los
        metrics = get_all_metrics(y_hat, y, self.task, self.los_info)
        # for k, v in metrics.items(): self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
            pd.to_pickle(self.model, self.checkpoint_path)
        return main_score
    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch: Batch of data.
            batch_idx (int): Index of the current batch.

        Returns:
            test_performance: Test performance metrics.
        """
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        self.model = pd.read_pickle(self.checkpoint_path)
        y_hat = self.model.predict(x)
        self.test_performance = get_all_metrics(y_hat, y, self.task, self.los_info)
        self.test_outputs = {'preds': y_hat, 'labels': y}
        return self.test_performance
    def configure_optimizers(self):
        pass
