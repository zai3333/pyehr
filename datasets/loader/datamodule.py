import os

import lightning as L
import pandas as pd
import torch
import torch.utils.data as data


class EhrDataset(data.Dataset):
    """EHR dataset.

    Attributes:
        data (pd.DataFrame): EHR data
        label (pd.DataFrame): EHR label
        pid (pd.DataFrame): patient id
    """
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data = pd.read_pickle(os.path.join(data_path,f'{mode}_x.pkl'))
        self.label = pd.read_pickle(os.path.join(data_path,f'{mode}_y.pkl'))
        self.pid = pd.read_pickle(os.path.join(data_path,f'{mode}_pid.pkl'))

    def __len__(self):
        """Return the number of patients."""
        return len(self.label) # number of patients

    def __getitem__(self, index):
        """Return the data of a patient."""
        return self.data[index], self.label[index], self.pid[index]


class EhrDataModule(L.LightningDataModule):
    """EHR data module.

    Attributes:
        data_path (str): path to the data
        batch_size (int): batch size,default=32
    """
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: str):
        """Set up the data module.

        Initialize the dataset objects for the training set, validation set, or test set based on the given stage.
        For the stage "fit", it creates dataset objects for the training set and validation set.
        For the stage "test", it creates a dataset object for the test set.

        Args:
            stage (str): stage of the data module, "fit" or "test"
        """
        if stage=="fit":
            self.train_dataset = EhrDataset(self.data_path, mode="train")
            self.val_dataset = EhrDataset(self.data_path, mode='val')
        if stage=="test":
            self.test_dataset = EhrDataset(self.data_path, mode='test')

    def train_dataloader(self):
        """Return the training dataloader."""
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , collate_fn=self.pad_collate, num_workers=8)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def test_dataloader(self):
        """Return the test dataloader."""
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def pad_collate(self, batch):
        """Pad the sequences in the batch to the same length.

        Args:
            batch (list): a list of tuples, each tuple contains the feature data, label data, and patient ID data of a patient.

        Returns:
            the padded feature data, label data, length data, and patient ID data.
        """
        xx, yy, pid = zip(*batch)
        lens = torch.as_tensor([len(x) for x in xx])
        # convert to tensor
        xx = [torch.tensor(x) for x in xx]
        yy = [torch.tensor(y) for y in yy]
        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
        return xx_pad, yy_pad, lens, pid