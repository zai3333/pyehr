from catboost import CatBoostClassifier, CatBoostRegressor


class CatBoost():
    """
    CatBoost model.

    a wrapper for the CatBoost classifier and regressor models. It is used for training and predicting outcomes or
    lengths of stay (LOS) based on the specified task.

    Attributes:
        `model` (CatBoostClassifier or CatBoostRegressor): CatBoost model.
        `task` (str): Task to be performed by the model. It can be either 'outcome' or 'los'.
        `seed` (int): Random seed.
        `n_estimators` (int): Number of trees.
        `learning_rate` (float): Learning rate.
        `max_depth` (int): Depth of the trees.
    """
    def __init__(self, **params):
        """
        Args:
            `**params` (dict): Dictionary containing the parameters of the model.
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']
        max_depth = params['max_depth']
        if task == "outcome":
            self.model = CatBoostClassifier(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=None, silent=True, allow_writing_files=False, loss_function="CrossEntropy")
        elif task == "los":
            self.model = CatBoostRegressor(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=None, silent=True, allow_writing_files=False, loss_function="RMSE")
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

    def fit(self, x, y):
        """fit the model.

        It is used to train the CatBoost model.The model is fitted based on the specified task: for the "outcome" task,
        it fits a CatBoostClassifier to predict class probabilities for the positive outcome; for the "los" task, it
        fits a CatBoostRegressor to predict the LOS.

        Args:
            `x` (numpy.ndarray): Features.
            `y` (numpy.ndarray): Labels.

        Raises:
            ValueError: If the task is neither "outcome" nor "los".
        """
        if self.task == "outcome":
            self.model.fit(x, y[:, 0])
        elif self.task == "los":
            self.model.fit(x, y[:, 1])
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    def predict(self, x):
        """predict the outcome or LOS.

        Args:
            `x` (numpy.ndarray): Features.

        Returns:
            If the task is "outcome", it returns the predicted class probabilities for the positive outcome.
            If the task is "los", it returns the predicted LOS.

        Raises:
            ValueError: If the task is neither "outcome" nor "los".
        """
        if self.task == "outcome":
            return self.model.predict_proba(x)[:, 1]
        elif self.task == "los":
            return self.model.predict(x)
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
