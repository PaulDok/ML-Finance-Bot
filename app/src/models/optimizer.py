import logging

import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger()


class ModelOptimizer:
    """
    A class which incapsulates hyperparameter tuning for a trading ML model
    """

    def __init__(
        self,
        model_type,
        param_dict: dict,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        X_val_scaled,
        y_val,
        early_stopping_rounds=50,
        # verbose=0,
        n_trials=20,
    ) -> None:
        self.model_type = model_type
        self.param_dict = param_dict

        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.X_val_scaled = X_val_scaled
        self.y_val = y_val
        self.early_stopping_rounds = early_stopping_rounds
        # self.verbose = verbose

        self.n_trials = n_trials

    def objective(self, trial) -> float:
        # 1) Parse params
        suggested_param = {}  # param storage

        # int
        int_params = self.param_dict.get("int", None)
        if int_params is not None:
            for k, v in int_params.items():
                suggested = trial.suggest_int(k, v["low"], v["high"])
                suggested_param.update({k: suggested})

        # log
        loguniform_params = self.param_dict.get("loguniform", None)
        if loguniform_params is not None:
            for k, v in loguniform_params.items():
                suggested = trial.suggest_float(k, v["low"], v["high"], log=True)
                suggested_param.update({k: suggested})

        # float
        float_params = self.param_dict.get("float", None)
        if float_params is not None:
            for k, v in float_params.items():
                suggested = trial.suggest_float(k, v["low"], v["high"])
                suggested_param.update({k: suggested})

        # constant
        const_params = self.param_dict.get("const", None)
        if const_params is not None:
            for k, v in const_params.items():
                suggested_param.update({k: v})

        # Инициализация модели с заданными параметрами
        model = self.model_type(**suggested_param)

        # Обучение модели с валидационной выборкой
        if self.param_dict.get("use_eval_set", None):
            _ = model.fit(
                self.X_train_scaled,
                self.y_train,
                eval_set=(self.X_test_scaled, self.y_test),
            )
        else:
            _ = model.fit(
                self.X_train_scaled,
                self.y_train,
            )

        # Предсказания на тренировочной и тестовой выборках
        y_train_pred_prob = model.predict_proba(self.X_train_scaled)[:, 1]
        y_test_pred_prob = model.predict_proba(self.X_test_scaled)[:, 1]

        # Оценка метрики ROC AUC на тренировочной и валидационной выборках
        train_roc_auc = roc_auc_score(self.y_train, y_train_pred_prob)
        test_roc_auc = roc_auc_score(self.y_test, y_test_pred_prob)

        # Штраф за переобучение (разница между тренировочной и валидационной метриками)
        overfitting_penalty = abs(train_roc_auc - test_roc_auc)

        # Целевая функция с учетом штрафа: при переобучении функция уменьшится
        score = test_roc_auc - overfitting_penalty

        return score

    def optimize(self):
        """
        Run hyperparameter tuning using Optuna
        """
        # 1. Создаем объект исследования и проводим его
        logger.info("Searching for best hyperparameters using Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1)

        # Получаем лучшие параметры и результат
        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший скорректированный ROC AUC Score: {best_score}")

        # 2. Fit model on train dataset, using X_test/y_test as eval_set for early stopping
        logger.info("Fitting model with best parameters...")
        model = self.model_type(
            **best_params,
            verbose=0,  # limit logging
        )
        if self.param_dict.get("use_eval_set", None):
            # Добавляем валидационную выборку как eval_set и используем early_stopping
            _ = model.fit(
                self.X_train_scaled,
                self.y_train,
                eval_set=(self.X_test_scaled, self.y_test),
                early_stopping_rounds=self.early_stopping_rounds,
            )
        else:
            # Просто фит
            _ = model.fit(
                self.X_train_scaled,
                self.y_train,
            )

        # 3. Generate predictions
        logger.info("Generating predictions...")
        y_train_pred_prob = model.predict_proba(self.X_train_scaled)[:, 1]
        y_test_pred_prob = model.predict_proba(self.X_test_scaled)[:, 1]
        y_val_pred_prob = model.predict_proba(self.X_val_scaled)[:, 1]

        # 4. Calculate ROC AUC
        logger.info("Evaluating ROC AUC...")
        train_roc_auc = roc_auc_score(self.y_train, y_train_pred_prob)
        test_roc_auc = roc_auc_score(self.y_test, y_test_pred_prob)
        val_roc_auc = roc_auc_score(self.y_val, y_val_pred_prob)

        # 5. Calculate metrics by threshold
        logger.info("Evaluating metrics by threshold...")
        train_metrics_table = calculate_metrics_table(self.y_train, y_train_pred_prob)
        test_metrics_table = calculate_metrics_table(self.y_test, y_test_pred_prob)
        val_metrics_table = calculate_metrics_table(self.y_val, y_val_pred_prob)

        # Display results
        results = {
            "TRAIN": (train_roc_auc, train_metrics_table),
            "TEST": (test_roc_auc, test_metrics_table),
            "VAL": (val_roc_auc, val_metrics_table),
        }
        for name, (roc_auc, metrics_table) in results.items():
            display_metrics_set(name, roc_auc, metrics_table)

        # Return the model and metrics
        return (
            model,
            best_params,
            (train_roc_auc, test_roc_auc, val_roc_auc),
            (train_metrics_table, test_metrics_table, val_metrics_table),
        )


# Метрики рассчитанные по трешхолдам
def calculate_metrics_table(
    y_true, y_pred_prob, thresholds=[0.5, 0.6, 0.7, 0.8]
) -> pd.DataFrame:
    """
    Evaluate Precision / Recall / Accuracy / F1-Score on different thresholds
    """
    metrics_table = []

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        metrics = {
            "Cutoff": threshold,
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }
        metrics_table.append(metrics)

    return pd.DataFrame(metrics_table) * 100


# Выводим ROC AUC и метрики для всех выборок
def display_metrics_set(name, roc_auc, metrics_table):
    print(f"\n=== Метрики для {name} выборки ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(metrics_table)
