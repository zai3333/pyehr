import os

import pandas as pd


def get_los_info(dataset_dir):
    """Get LOS information from the dataset directory."""
    path = os.path.join(dataset_dir, 'los_info.pkl')
    los_info = pd.read_pickle(path)
    return los_info
