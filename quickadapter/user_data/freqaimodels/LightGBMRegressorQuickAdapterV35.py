import logging
from typing import Any, Dict

from lightgbm import LGBMRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import pandas as pd
import scipy as spy
import optuna
import sklearn
import warnings
import re

N_TRIALS = 36
TEST_SIZE = 0.1

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class LightGBMRegressorQuickAdapterV35(BaseRegressionModel):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__optuna_config = self.freqai_info.get("optuna_hyperopt", {})
        self.__optuna_hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.__optuna_config.get("enabled", False)
            and self.data_split_parameters.get("test_size", TEST_SIZE) > 0
        )
        self.__optuna_hp = {}

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"]
        test_weights = data_dictionary["test_weights"]

        init_model = self.get_init_model(dk.pair)

        model = LGBMRegressor(**self.model_training_parameters)

        start = time.time()
        if self.__optuna_hyperopt:
            study_name = str(dk.pair)
            storage_dir = str(dk.full_path)
            pruner = optuna.pruners.HyperbandPruner()
            study = optuna.create_study(
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(
                    multivariate=True,
                    group=True,
                ),
                pruner=pruner,
                direction=optuna.study.StudyDirection.MINIMIZE,
                storage=optuna.storages.JournalStorage(
                    optuna.storages.journal.JournalFileBackend(
                        f"{storage_dir}/optuna-{sanitize_path(study_name)}.log"
                    )
                ),
                load_if_exists=True,
            )
            study.optimize(
                lambda trial: objective(
                    trial,
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.data_split_parameters.get("test_size", TEST_SIZE),
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self.__optuna_config.get("candles_step", 100),
                    self.model_training_parameters,
                ),
                n_trials=self.__optuna_config.get("n_trials", N_TRIALS),
                n_jobs=self.__optuna_config.get("n_jobs", 1),
                timeout=self.__optuna_config.get("timeout", 3600),
                gc_after_trial=True,
            )

            self.__optuna_hp = study.best_params
            # log params
            for key, value in self.__optuna_hp.items():
                logger.info(f"Optuna hyperopt | {key:>20s} : {value}")
            logger.info(
                f"Optuna hyperopt | {'best objective value':>20s} : {study.best_value}"
            )

            train_window = self.__optuna_hp.get("train_period_candles")
            X = X.tail(train_window)
            y = y.tail(train_window)
            train_weights = train_weights[-train_window:]

            test_window = self.__optuna_hp.get("test_period_candles")
            X_test = X_test.tail(test_window)
            y_test = y_test.tail(test_window)
            test_weights = test_weights[-test_window:]

            if dk.pair not in self.freqai_info["feature_parameters"]:
                self.freqai_info["feature_parameters"][dk.pair] = {}
            self.freqai_info["feature_parameters"][dk.pair]["label_period_candles"] = (
                self.__optuna_hp.get("label_period_candles")
            )

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            init_model=init_model,
        )
        time_spent = time.time() - start
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        warmed_up = True

        num_candles = self.freqai_info.get("fit_live_predictions_candles", 100)
        if self.live:
            if not hasattr(self, "exchange_candles"):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candle_diff = len(self.dd.historic_predictions[pair].index) - (
                num_candles + self.exchange_candles
            )
            if candle_diff < 0:
                logger.warning(
                    f"{pair}: fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go"
                )
                warmed_up = False

        pred_df_full = (
            self.dd.historic_predictions[pair].tail(num_candles).reset_index(drop=True)
        )

        if not warmed_up:
            dk.data["extra_returns_per_train"]["&s-maxima_sort_threshold"] = 2
            dk.data["extra_returns_per_train"]["&s-minima_sort_threshold"] = -2
        else:
            if (
                self.__optuna_hyperopt
                and pair in self.freqai_info["feature_parameters"]
            ):
                label_period_candles = self.__optuna_hp.get(
                    "label_period_candles", self.ft_params["label_period_candles"]
                )
            else:
                label_period_candles = self.ft_params["label_period_candles"]
            min_pred, max_pred = min_max_pred(
                pred_df_full,
                num_candles,
                label_period_candles,
            )
            dk.data["extra_returns_per_train"]["&s-maxima_sort_threshold"] = max_pred[
                "&s-extrema"
            ]
            dk.data["extra_returns_per_train"]["&s-minima_sort_threshold"] = min_pred[
                "&s-extrema"
            ]

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for ft in dk.label_list:
            # f = spy.stats.norm.fit(pred_df_full[ft])
            dk.data["labels_std"][ft] = 0  # f[1]
            dk.data["labels_mean"][ft] = 0  # f[0]

        # fit the DI_threshold
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            di_values = pd.to_numeric(pred_df_full["DI_values"], errors="coerce")
            di_values = di_values.dropna()
            f = spy.stats.weibull_min.fit(di_values)
            cutoff = spy.stats.weibull_min.ppf(
                self.freqai_info.get("outlier_threshold", 0.999), *f
            )

        dk.data["DI_value_mean"] = pred_df_full["DI_values"].mean()
        dk.data["DI_value_std"] = pred_df_full["DI_values"].std()
        dk.data["extra_returns_per_train"]["DI_value_param1"] = f[0]
        dk.data["extra_returns_per_train"]["DI_value_param2"] = f[1]
        dk.data["extra_returns_per_train"]["DI_value_param3"] = f[2]
        dk.data["extra_returns_per_train"]["DI_cutoff"] = cutoff

    def eval_set_and_weights(self, X_test, y_test, test_weights):
        if self.data_split_parameters.get("test_size", TEST_SIZE) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(X_test, y_test)]
            eval_weights = [test_weights]

        return eval_set, eval_weights


def min_max_pred(
    pred_df: pd.DataFrame, fit_live_predictions_candles: int, label_period_candles: int
):
    pred_df_sorted = pd.DataFrame()
    for label in pred_df.keys():
        if pred_df[label].dtype == object:
            continue
        pred_df_sorted[label] = pred_df[label]

    for col in pred_df_sorted:
        pred_df_sorted[col] = pred_df_sorted[col].sort_values(
            ascending=False, ignore_index=True
        )
    frequency = fit_live_predictions_candles / (label_period_candles * 2)
    max_pred = pred_df_sorted.iloc[: int(frequency)].mean()
    min_pred = pred_df_sorted.iloc[-int(frequency) :].mean()
    return min_pred, max_pred


def objective(
    trial,
    X,
    y,
    train_weights,
    X_test,
    y_test,
    test_weights,
    test_size,
    fit_live_predictions_candles,
    candles_step,
    params,
):
    min_train_window: int = 600
    max_train_window: int = (
        len(X) if len(X) > min_train_window else (min_train_window + len(X))
    )
    train_window = trial.suggest_int(
        "train_period_candles", min_train_window, max_train_window, step=candles_step
    )
    X = X.tail(train_window)
    y = y.tail(train_window)
    train_weights = train_weights[-train_window:]

    min_test_window: int = int(min_train_window * test_size)
    max_test_window: int = (
        len(X_test)
        if len(X_test) > min_test_window
        else (min_test_window + len(X_test))
    )
    test_window = trial.suggest_int(
        "test_period_candles", min_test_window, max_test_window, step=candles_step
    )
    X_test = X_test.tail(test_window)
    y_test = y_test.tail(test_window)
    test_weights = test_weights[-test_window:]

    # Fit the model
    model = LGBMRegressor(objective="rmse", **params)
    model.fit(
        X=X,
        y=y,
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)],
        eval_sample_weight=[test_weights],
        eval_metric="rmse",
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, "rmse")],
    )
    y_pred = model.predict(X_test)

    min_label_period_candles = int(fit_live_predictions_candles / 10)
    max_label_period_candles = int(fit_live_predictions_candles / 2)
    label_period_candles = trial.suggest_int(
        "label_period_candles",
        min_label_period_candles,
        max_label_period_candles,
    )
    y_test = y_test.tail(label_period_candles)
    y_pred = y_pred[-label_period_candles:]

    error = sklearn.metrics.root_mean_squared_error(y_test, y_pred)

    return error


def hp_objective(trial, X, y, train_weights, X_test, y_test, test_weights, params):
    study_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    params = {**params, **study_params}

    # Fit the model
    model = LGBMRegressor(objective="rmse", **params)
    model.fit(
        X=X,
        y=y,
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)],
        eval_sample_weight=[test_weights],
        eval_metric="rmse",
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, "rmse")],
    )
    y_pred = model.predict(X_test)

    error = sklearn.metrics.root_mean_squared_error(y_test, y_pred)

    return error


def sanitize_path(path: str) -> str:
    allowed = re.compile(r"[^A-Za-z0-9 _\-\.\(\)]")
    return allowed.sub("_", path)
