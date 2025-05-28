import logging

from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from .lstm_model import LSTMModel

logger = logging.getLogger()


class EnsembleModel:
    """
    Wrapper for an Ensemble (VotingClassifier) of several ML models for next step prediction
    """

    def __init__(
        self,
        lr_l1_ratio=0.01,
        lr_solver="saga",
        lr_penalty="elasticnet",
        cb_iterations=100,
        cb_depth=3,
        cb_lr=0.01,
        cb_l2_leaf_reg=1e-3,
        cb_bagging_temperature=0,
        cb_rsm=0.5,
        cb_subsample=0.5,
        cb_random_seed=42,
        cb_early_stopping_rounds=50,
        lstm_window_size=30,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        lstm_batch_size=32,
        lstm_lr=0.001,
        lstm_num_epochs=10,
        verbose=0,
    ) -> None:
        # Ensemble components
        self.lr_model = LogisticRegression(
            **{"l1_ratio": lr_l1_ratio, "solver": lr_solver, "penalty": lr_penalty}
        )
        self.cb_model = CatBoostClassifier(
            **{
                "iterations": cb_iterations,
                "depth": cb_depth,
                "learning_rate": cb_lr,
                "l2_leaf_reg": cb_l2_leaf_reg,
                "bagging_temperature": cb_bagging_temperature,
                "rsm": cb_rsm,
                "subsample": cb_subsample,
                "random_seed": cb_random_seed,
                "verbose": verbose,
                "early_stopping_rounds": cb_early_stopping_rounds,
            }
        )
        self.lstm_model = LSTMModel(
            **{
                "window_size": lstm_window_size,
                "hidden_size": lstm_hidden_size,
                "num_layers": lstm_num_layers,
                "batch_size": lstm_batch_size,
                "lr": lstm_lr,
                "num_epochs": lstm_num_epochs,
                "verbose": verbose,
            }
        )
        self.voting_clf = VotingClassifier(
            estimators=[
                ("lr", self.lr_model),
                ("cb", self.cb_model),
                ("lstm", self.lstm_model),
            ],
            voting="soft",
        )

        self.verbose = verbose

    def fit(self, X_train, y_train):
        return self.voting_clf.fit(X_train, y_train)

    def predict_proba(self, X_raw):
        return self.voting_clf.predict_proba(X_raw)
