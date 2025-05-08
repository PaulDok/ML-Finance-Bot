import json
import logging
import uuid

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..core import utils
from ..models import optimizer
from ..strategy.ml_strategy import MLBasedStrategy

logger = logging.getLogger()


def ml_model_strategy_training_loop(
    tickers: list[str],
    interval: str,
    train_start: str,
    train_end: str,
    test_end: str,
    valid_end: str,
    model_options: dict,
) -> None:
    """
    Full logic for ML-based strategy testing:
    - get data from DB
    - for each ticker
        - tune ML models' hyperparameters
        - run backtest using best model
        - concat technical and business metrics
        - save them to DB for further analysis
    """
    # Get preprocessed data
    data, features = utils.get_preprocessed_history(
        tickers=tickers, start=train_start, end=valid_end, interval=interval
    )

    # Add feature column
    data = utils.add_target(data)

    # Loop over tickers in dataset
    grand_result = []
    for ticker in data["Ticker"].unique():
        logger.info(f"~ ~ ~ Modelling for {ticker} ~ ~ ~")
        ticker_data = data[data["Ticker"] == ticker].reset_index(drop=True)

        # Split dataset into parts
        X_train, y_train, X_test, y_test, X_val, y_val = utils.train_test_valid_split(
            ticker_data,
            train_start=train_start,
            train_end=train_end,
            test_end=test_end,
            valid_end=valid_end,
            drop_leaky=True,
            target_col="target",
        )
        logger.info(
            f"{X_train.shape=} | {y_train.shape=} || {X_test.shape=} | {y_test.shape=} || {X_val.shape=} | {y_val.shape=}"
        )

        # Scale train / test / validation datasets - fit on train
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[features])
        X_test_scaled = scaler.transform(X_test[features])
        X_val_scaled = scaler.transform(X_val[features])

        # Для каждого потенциального типа модели:
        for model_type, param_dict in model_options.items():
            logger.info(f"~ ~ Iteration for {model_type.__name__} ~ ~")
            # Тюним гиперпараметры на train / test датасетах, выбираем лучшее
            model_optimizer = optimizer.ModelOptimizer(
                model_type,
                param_dict,
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                X_val_scaled,
                y_val,
            )
            (
                model,
                best_params,
                (train_roc_auc, test_roc_auc, val_roc_auc),
                (train_metrics_table, test_metrics_table, val_metrics_table),
            ) = model_optimizer.optimize()

            # Run backtesting and collect all metrics
            logger.info("Running backtesting and collecting all metrics..")
            all_parts_metrics = []
            experiment_id = str(uuid.uuid1())
            for (
                X_data,
                X_scaled_data,
                metrics_table,
                roc_auc,
                run_type,
                start_dt,
                end_dt,
            ) in [
                (
                    X_train,
                    X_train_scaled,
                    train_metrics_table,
                    train_roc_auc,
                    "TRAIN",
                    train_start,
                    train_end,
                ),
                (
                    X_test,
                    X_test_scaled,
                    test_metrics_table,
                    test_roc_auc,
                    "TEST",
                    train_end,
                    test_end,
                ),
                (
                    X_val,
                    X_val_scaled,
                    val_metrics_table,
                    val_roc_auc,
                    "VALID",
                    test_end,
                    valid_end,
                ),
            ]:
                # Evaluate business metrics using Backtesting
                all_threshold_stats = []
                for threshold in [0.5, 0.6, 0.7, 0.8]:
                    bt_stats = backtest_ml_strategy(
                        model, ticker_data, X_data, X_scaled_data, threshold
                    )
                    bt_stats = pd.DataFrame(bt_stats).transpose()
                    bt_stats["Cutoff"] = threshold * 100
                    all_threshold_stats.append(bt_stats)
                all_threshold_stats = pd.concat(
                    all_threshold_stats, axis=0
                ).reset_index(drop=True)

                # Concat with technical ones
                metrics_table["ROC_AUC"] = roc_auc
                metrics_table = pd.merge(
                    metrics_table, all_threshold_stats, how="left", on="Cutoff"
                )

                # Add general data
                general_data = pd.DataFrame(
                    {
                        "Experiment_ID": experiment_id,
                        "Ticker": ticker,
                        "Interval": interval,
                        "Type": run_type,
                        "START_DT": start_dt,
                        "END_DT": end_dt,
                        "Model": model.__class__.__name__,
                        "Model_params": json.dumps(best_params),
                    },
                    index=metrics_table.index,
                )
                metrics_table = pd.concat([general_data, metrics_table], axis=1)

                all_parts_metrics.append(metrics_table)

            # All metrics (Train + Test + Valid)
            all_parts_metrics = pd.concat(all_parts_metrics, axis=0).reset_index(
                drop=True
            )

            # Save to DB
            logger.info("Saving results to DB...")
            # Make sure the table exists
            utils.create_experiment_results_table()
            # Rename columns to valid names for DB
            all_parts_metrics.columns = [
                col.replace("-", "_")
                .replace("#", "Num")
                .replace(" ", "_")
                .replace("[%]", "pct")
                for col in all_parts_metrics.columns
            ]
            # Upload experiment results to DB
            utils.upload_experiment_results_to_sqlite(all_parts_metrics)

            grand_result.append(all_parts_metrics)

    grand_result = pd.concat(grand_result, axis=0).reset_index(drop=True)
    logger.info("ML Model Strategy training loop complete!")
    return grand_result


def backtest_ml_strategy(
    model,
    ticker_data: pd.DataFrame,
    X_data: pd.DataFrame,
    X_scaled_data: pd.DataFrame,
    threshold: float,
):
    # Run a backtest
    stats, bt = utils.backtest_strategy(
        strategy_class=MLBasedStrategy,
        y_test=pd.merge(ticker_data, X_data["Date"], on="Date", how="inner")[
            ["Date", "Close"]
        ],
        X_test=pd.merge(ticker_data, X_data["Date"], on="Date", how="inner")[
            ["Open", "High", "Low", "Volume"]
        ],
        # X_test=pd.DataFrame(X_train_scaled, columns=features),
        strategy_params={
            "model": model,
            "threshold": threshold,
            "y_pred_prob": model.predict_proba(X_scaled_data)[:, 1],
        },
    )

    # Return most
    return stats[["Return [%]", "Win Rate [%]", "# Trades"]]
