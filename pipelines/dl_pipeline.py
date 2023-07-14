import os

import lightning as L
import torch
import torch.nn as nn

import models
from datasets.loader.unpad import unpad_y
from losses import get_loss
from metrics import get_all_metrics, check_metric_is_better
from models.utils import generate_mask, get_last_visit


class DlPipeline(L.LightningModule):
    """A PyTorch Lightning module for a deep learning pipeline.

    Attributes:
        demo_dim: Dimension of the demographic features.
        lab_dim: Dimension of the lab features.
        input_dim: Total input dimension.
        hidden_dim: Dimension of the hidden layers.
        output_dim: Dimension of the output layer.
        learning_rate: Learning rate for the optimizer.
        task: Task type.
        los_info: Information about LOS.
        model_name: Name of the model.
        main_metric: Main metric used for evaluation.
        time_aware: Whether the model is time-aware. Default: False.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.demo_dim = config["demo_dim"]
        self.lab_dim = config["lab_dim"]
        self.input_dim = self.demo_dim + self.lab_dim
        config["input_dim"] = self.input_dim
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.task = config["task"]
        self.los_info = config["los_info"]
        self.model_name = config["model"]
        self.main_metric = config["main_metric"]
        self.time_aware = config.get("time_aware", False)
        self.cur_best_performance = {}
        self.embedding: torch.Tensor

        if self.model_name == "StageNet":
            config["chunk_size"] = self.hidden_dim

        model_class = getattr(models, self.model_name)
        self.ehr_encoder = model_class(**config)
        if self.task == "outcome":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif self.task == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif self.task == "multitask":
            self.head = models.heads.MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_performance = {}
        self.test_outputs = {}

    def forward(self, x, lens):
        """Forward pass.

        Args:
            x: Input tensor.
            lens: Length of each sequence.

        Returns:
            y_hat: Predicted output.
            embedding: Embedding of the input.
        """
        if self.model_name == "ConCare":
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding, decov_loss = self.ehr_encoder(x_lab, x_demo, mask)
            embedding, decov_loss = embedding.to(x.device), decov_loss.to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding, decov_loss
        elif self.model_name in ["GRASP", "Agent"]:
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding = self.ehr_encoder(x_lab, x_demo, mask).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["AdaCare", "RETAIN", "TCN", "Transformer", "StageNet"]:
            mask = generate_mask(lens)
            embedding = self.ehr_encoder(x, mask).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["GRU", "LSTM", "RNN", "MLP"]:
            embedding = self.ehr_encoder(x).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["MCGRU"]:
            x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
            embedding = self.ehr_encoder(x_lab, x_demo).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding

    def _get_loss(self, x, y, lens):
        """Get loss
        Args:
            x: Input tensor.
            y: Target tensor.
            lens: Length of each sequence.

        Returns:
            loss: Loss value.
            y: Target tensor.
            y_hat: Predicted output.
        """
        if self.model_name == "ConCare":
            y_hat, embedding, decov_loss = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
            loss += 10*decov_loss
        else:
            y_hat, embedding = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
        return loss, y, y_hat
    def training_step(self, batch, batch_idx):
        """Defines the training step for the model.

        Args:
            batch (tuple): A tuple containing the batch data (x, y, lens, pid) where:
                x (torch.Tensor): Input data tensor of shape (batch_size, sequence_length, input_dim).
                y (torch.Tensor): Target labels tensor of shape (batch_size, sequence_length).
                lens (torch.Tensor): Sequence lengths tensor of shape (batch_size,).
                pid (torch.Tensor): Patient IDs tensor of shape (batch_size,).
            batch_idx (int): Index of the batch.

        Returns:
            loss: Loss value.
        """
        x, y, lens, pid = batch
        loss, y, y_hat = self._get_loss(x, y, lens)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        """Defines the validation step for the model.

        Args:
            batch (tuple): A tuple containing the batch data (x, y, lens, pid) where:
                x (torch.Tensor): Input data tensor of shape (batch_size, sequence_length, input_dim).
                y (torch.Tensor): Target labels tensor of shape (batch_size, sequence_length).
                lens (torch.Tensor): Sequence lengths tensor of shape (batch_size,).
                pid (torch.Tensor): Patient IDs tensor of shape (batch_size,).
            batch_idx (int): Index of the batch.

        Returns:
            loss: Loss value.
        """
        x, y, lens, pid = batch
        loss, y, y_hat = self._get_loss(x, y, lens)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss
    def on_validation_epoch_end(self):
        """Performs operations at the end of each validation epoch, calculates val_loss , all metrics and main_score."""
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)
        metrics = get_all_metrics(y_pred, y_true, self.task, self.los_info)
        for k, v in metrics.items(): self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        """Defines the test step for the model.

        Args:
            batch (tuple): A tuple containing the batch data (x, y, lens, pid) where:
                x (torch.Tensor): Input data tensor of shape (batch_size, sequence_length, input_dim).
                y (torch.Tensor): Target labels tensor of shape (batch_size, sequence_length).
                lens (torch.Tensor): Sequence lengths tensor of shape (batch_size,).
                pid (torch.Tensor): Patient IDs tensor of shape (batch_size,).
            batch_idx (int): Index of the batch.

        Returns:
            loss: Loss value.
        """
        x, y, lens, pid = batch
        loss, y, y_hat = self._get_loss(x, y, lens)
        outs = {'y_pred': y_hat, 'y_true': y, 'lens': lens}
        self.test_step_outputs.append(outs)
        return loss
    def on_test_epoch_end(self):
        """Performs operations at the end of each test epoch, get test_performance."""
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).detach().cpu()
        lens = torch.cat([x['lens'] for x in self.test_step_outputs]).detach().cpu()
        self.test_performance = get_all_metrics(y_pred, y_true, self.task, self.los_info)
        self.test_outputs = {'preds': y_pred, 'labels': y_true, 'lens': lens}
        self.test_step_outputs.clear()
        return self.test_performance

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer