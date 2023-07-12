import torch
from torch import nn


class TimeAwareLoss(nn.Module):
    """Time-aware loss function.

    Attributes:
        decay_rate: Decay rate of the exponential decay.
        reward_factor: Factor of the reward term.
        bce: Binary cross entropy.
        decay_rate: Decay rate of los decay.
        reward_factor: Factor of the reward term.
    """
    def __init__(self, decay_rate=0.1, reward_factor=0.1):
        super(TimeAwareLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.decay_rate = decay_rate
        self.reward_factor = reward_factor

    def forward(self, outcome_pred, outcome_true, los_true):
        """Calculates the time-aware loss.
        Args:
            outcome_pred: Predicted outcome.
            outcome_true: True outcome.
            los_true: True length of stay.

        Returns:
            Time-aware loss.
        """
        los_weights = torch.exp(-self.decay_rate * los_true)  # Exponential decay
        loss_unreduced = self.bce(outcome_pred, outcome_true)

        reward_term = (los_true * torch.abs(outcome_true - outcome_pred)).mean()  # Reward term
        loss = (loss_unreduced * los_weights).mean()-self.reward_factor * reward_term  # Weighted loss
        
        return torch.clamp(loss, min=0)

def get_time_aware_loss(outcome_pred, outcome_true, los_true):
    """Calculates the time-aware loss.

    Encapsulates the computation of the time aware loss into this function.
    """
    time_aware_loss = TimeAwareLoss()
    return time_aware_loss(outcome_pred, outcome_true, los_true)

if __name__ == "__main__":
    outcome_pred = torch.tensor([0.1])
    outcome_true = torch.tensor([1.])
    los_true = torch.tensor([-4.0])
    print(get_time_aware_loss(outcome_pred, outcome_true, los_true))
