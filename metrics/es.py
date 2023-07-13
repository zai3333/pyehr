import numpy as np


def calculate_confusion_matrix_value_result(outcome_pred, outcome_true):
    """Calculate confusion matrix value result

    Args:
        outcome_pred (float): predicted outcome
        outcome_true (float): true outcome

    Returns:
        str: confusion matrix value result

    Raises:
        ValueError: Unknown value occurred
    """
    outcome_pred = 1 if outcome_pred > 0.5 else 0
    if outcome_pred == 1 and outcome_true == 1:
        return "tp"
    elif outcome_pred == 0 and outcome_true == 0:
        return "tn"
    elif outcome_pred == 1 and outcome_true == 0:
        return "fp"
    elif outcome_pred == 0 and outcome_true == 1:
        return "fn"
    else:
        raise ValueError("Unknown value occurred")

def calculate_es(los_true, threshold, case="tp"):
    """Calculate mortality score

    Args:
        los_true (float): true los
        threshold (float): threshold
        case (str, optional): case. Defaults to "tp".

    Returns:
        float: early mortality score
    """
    metric = 0.0
    if case == "tp":
        if los_true >= threshold:  # predict correct in early stage
            metric = 1
        else:
            metric = los_true / threshold
    elif case == "fn":
        if los_true >= threshold:  # predict wrong in early stage
            metric = 0
        else:
            metric = los_true / threshold - 1
    elif case == "tn":
        metric = 0.0
    elif case == "fp":
        metric = -0.1 # penalty term
    return metric


def es_score(
    y_true_outcome,
    y_true_los,
    y_pred_outcome,
    threshold,
    verbose=0
):
    """Calculates the ES for a set of predictions.

    Args:
        y_true_outcome (List[float]): List of true outcome labels.
        y_true_los (List[float]): List of true length of stay values.
        y_pred_outcome (List[float]): List of predicted outcome labels.
        threshold (float): Threshold value for determining early prediction.
        verbose (int, optional): Verbosity level. Set to 1 to print the ES score. Defaults to 0.

    Returns:
        Dict[str, float]: Dictionary containing the ES score.

    Note:
        - y/predictions are already flattened here
        - so we don't need to consider visits_length
    """
    metric = []
    metric_optimal = []
    num_records = len(y_pred_outcome)
    for i in range(num_records):
        cur_outcome_pred = y_pred_outcome[i]
        cur_outcome_true = y_true_outcome[i]
        cur_los_true = y_true_los[i]
        prediction_result = calculate_confusion_matrix_value_result(cur_outcome_pred, cur_outcome_true)
        prediction_result_optimal = calculate_confusion_matrix_value_result(cur_outcome_true, cur_outcome_true)
        metric.append(
            calculate_es(
                cur_los_true,
                threshold,
                case=prediction_result,
            )
        )
        metric_optimal.append(
            calculate_es(
                cur_los_true,
                threshold,
                case=prediction_result_optimal,
            )
        )
    metric = np.array(metric)
    metric_optimal = np.array(metric_optimal)
    result = 0.0
    if metric_optimal.sum() > 0.0:
        result = metric.sum() / metric_optimal.sum()
    result = max(result, -1.0)
    if verbose:
        print("ES Score:", result)
    if isinstance(result, np.float64):
        result = result.item()
    return {"es": result}

if __name__ == "__main__":
    y_true_outcome = np.array([0,1])
    y_true_los = np.array([5,5])
    y_pred_outcome = np.array([0.7,0.7])
    y_pred_los = np.array([10,10])
    large_los = 110
    threshold = 10
    print(es_score(
        y_true_outcome,
        y_true_los,
        y_pred_outcome,
        threshold,
        verbose=0
    ))