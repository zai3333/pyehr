experiments_configs = [
 {'model': 'RNN',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'GRU',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'TCN',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'StageNet',
  'dataset': 'tjh',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'RNN',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'GRU',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'TCN',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1,
  'time_aware': True,
 },
 {'model': 'StageNet',
  'dataset': 'cdsl',
  'task': 'outcome',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1,
  'time_aware': True,
 },
    {
        "model": "RETAIN",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "RETAIN",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "Agent",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "Agent",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "AdaCare",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "AdaCare",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
        'time_aware': True,
    },
    {
        "model": "MLP",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "MLP",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "Transformer",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "Transformer",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "LSTM",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 128,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "LSTM",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "GRASP",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 128,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "GRASP",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
        "time_aware": True,
    },
    {
        "model": "MCGRU",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
        "time_aware": True
    },
    {
        "model": "MCGRU",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
        "time_aware": True
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "outcome",
        "dataset": "tjh",
        "demo_dim": 2,
        "lab_dim": 73,
        "batch_size": 64,
        "main_metric": "auprc",
        "model": "MCGRU"
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "outcome",
        "dataset": "cdsl",
        "demo_dim": 2,
        "lab_dim": 97,
        "batch_size": 128,
        "main_metric": "auprc",
        "model": "MCGRU"
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "los",
        "dataset": "tjh",
        "demo_dim": 2,
        "lab_dim": 73,
        "batch_size": 64,
        "main_metric": "mae",
        "model": "MCGRU"
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "los",
        "dataset": "cdsl",
        "demo_dim": 2,
        "lab_dim": 97,
        "batch_size": 128,
        "main_metric": "mae",
        "model": "MCGRU"
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "multitask",
        "dataset": "tjh",
        "demo_dim": 2,
        "lab_dim": 73,
        "batch_size": 64,
        "main_metric": "auprc",
        "model": "MCGRU"
    },
    {
        "epochs": 100,
        "patience": 10,
        "learning_rate": 0.001,
        "output_dim": 1,
        "hidden_dim": 64,
        "task": "multitask",
        "dataset": "cdsl",
        "demo_dim": 2,
        "lab_dim": 97,
        "batch_size": 128,
        "main_metric": "auprc",
        "model": "MCGRU"
    }
]