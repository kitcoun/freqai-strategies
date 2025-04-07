import logging
import json
import time
import numpy as np
import pandas as pd
import scipy as sp
import optuna
import sklearn
import warnings

from functools import cached_property
from typing import Any, Callable, Optional
from pathlib import Path
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


TEST_SIZE = 0.1

EXTREMA_COLUMN = "&s-extrema"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class QuickAdapterRegressorV3(BaseRegressionModel):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This freqaimodel is experimental (as with all models released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    version = "3.6.8"

    @cached_property
    def __optuna_config(self) -> dict:
        optuna_default_config = {
            "enabled": False,
            "n_jobs": min(
                self.freqai_info.get("optuna_hyperopt", {}).get("n_jobs", 1),
                max(int(self.max_system_threads / 4), 1),
            ),
            "storage": "file",
            "continuous": True,
            "warm_start": True,
            "n_trials": 36,
            "timeout": 7200,
            "candles_step": 10,
        }
        return {
            **optuna_default_config,
            **self.freqai_info.get("optuna_hyperopt", {}),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pairs = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI model requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            self.freqai_info.get("identifier") is None
            or self.freqai_info.get("identifier").strip() == ""
        ):
            raise ValueError(
                "FreqAI model requires 'identifier' defined in the freqai section configuration"
            )
        self.__optuna_hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.__optuna_config.get("enabled")
            and self.data_split_parameters.get("test_size", TEST_SIZE) > 0
        )
        self.__optuna_hp_rmse: dict[str, float] = {}
        self.__optuna_period_rmse: dict[str, float] = {}
        self.__optuna_hp_params: dict[str, dict] = {}
        self.__optuna_period_params: dict[str, dict] = {}
        for pair in self.pairs:
            self.__optuna_hp_rmse[pair] = -1
            self.__optuna_period_rmse[pair] = -1
            self.__optuna_hp_params[pair] = (
                self.optuna_load_best_params(pair, "hp") or {}
            )
            self.__optuna_period_params[pair] = (
                self.optuna_load_best_params(pair, "period") or {}
            )
        logger.info(
            f"Initialized {self.__class__.__name__} {self.freqai_info.get('regressor', 'xgboost')} regressor model version {self.version}"
        )

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
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

        model_training_parameters = self.model_training_parameters

        init_model = self.get_init_model(dk.pair)

        start = time.time()
        if self.__optuna_hyperopt:
            self.optuna_optimize(
                dk.pair,
                "hp",
                lambda trial: hp_objective(
                    trial,
                    self.freqai_info.get("regressor", "xgboost"),
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    model_training_parameters,
                ),
                self.__optuna_hp_params,
                self.__optuna_hp_rmse,
            )

            if self.__optuna_hp_params.get(dk.pair):
                model_training_parameters = {
                    **model_training_parameters,
                    **self.__optuna_hp_params[dk.pair],
                }

            self.optuna_optimize(
                dk.pair,
                "period",
                lambda trial: period_objective(
                    trial,
                    self.freqai_info.get("regressor", "xgboost"),
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self.ft_params.get("label_period_candles", 50),
                    self.__optuna_config.get("candles_step"),
                    model_training_parameters,
                ),
                self.__optuna_period_params,
                self.__optuna_period_rmse,
            )

            if self.__optuna_period_params.get(dk.pair):
                train_window = self.__optuna_period_params[dk.pair].get(
                    "train_period_candles"
                )
                X = X.iloc[-train_window:]
                y = y.iloc[-train_window:]
                train_weights = train_weights[-train_window:]

                test_window = self.__optuna_period_params[dk.pair].get(
                    "test_period_candles"
                )
                X_test = X_test.iloc[-test_window:]
                y_test = y_test.iloc[-test_window:]
                test_weights = test_weights[-test_window:]

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        model = train_regressor(
            regressor=self.freqai_info.get("regressor", "xgboost"),
            X=X,
            y=y,
            train_weights=train_weights,
            eval_set=eval_set,
            eval_weights=eval_weights,
            model_training_parameters=model_training_parameters,
            init_model=init_model,
        )
        time_spent = time.time() - start
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def get_label_period_candles(self, pair: str) -> int:
        label_period_candles = self.__optuna_period_params.get(pair).get(
            "label_period_candles"
        )
        if label_period_candles:
            return label_period_candles
        return self.ft_params.get("label_period_candles", 50)

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        warmed_up = True

        num_candles = self.freqai_info.get("fit_live_predictions_candles", 100)
        if self.live:
            if not hasattr(self, "exchange_candles"):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candles_diff = len(self.dd.historic_predictions[pair].index) - (
                num_candles + self.exchange_candles
            )
            if candles_diff < 0:
                logger.warning(
                    f"{pair}: fit live predictions not warmed up yet. Still {abs(candles_diff)} candles to go."
                )
                warmed_up = False

        pred_df_full = (
            self.dd.historic_predictions[pair]
            .iloc[-num_candles:]
            .reset_index(drop=True)
        )

        if not warmed_up:
            dk.data["extra_returns_per_train"][MINIMA_THRESHOLD_COLUMN] = -2
            dk.data["extra_returns_per_train"][MAXIMA_THRESHOLD_COLUMN] = 2
        else:
            min_pred, max_pred = self.min_max_pred(pred_df_full)
            dk.data["extra_returns_per_train"][MINIMA_THRESHOLD_COLUMN] = min_pred
            dk.data["extra_returns_per_train"][MAXIMA_THRESHOLD_COLUMN] = max_pred

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label in dk.label_list + dk.unique_class_list:
            if pred_df_full[label].dtype == object:
                continue
            if not warmed_up:
                f = [0, 0]
            else:
                f = sp.stats.norm.fit(pred_df_full[label])
            dk.data["labels_mean"][label], dk.data["labels_std"][label] = f[0], f[1]

        # fit the DI_threshold
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            di_values = pd.to_numeric(pred_df_full["DI_values"], errors="coerce")
            di_values = di_values.dropna()
            f = sp.stats.weibull_min.fit(di_values)
            cutoff = sp.stats.weibull_min.ppf(
                self.freqai_info.get("outlier_threshold", 0.999), *f
            )

        dk.data["DI_value_mean"] = pred_df_full["DI_values"].mean()
        dk.data["DI_value_std"] = pred_df_full["DI_values"].std()
        dk.data["extra_returns_per_train"]["DI_value_param1"] = f[0]
        dk.data["extra_returns_per_train"]["DI_value_param2"] = f[1]
        dk.data["extra_returns_per_train"]["DI_value_param3"] = f[2]
        dk.data["extra_returns_per_train"]["DI_cutoff"] = cutoff

        dk.data["extra_returns_per_train"]["label_period_candles"] = (
            self.get_label_period_candles(pair)
        )
        dk.data["extra_returns_per_train"]["hp_rmse"] = self.__optuna_hp_rmse.get(
            pair, -1
        )
        dk.data["extra_returns_per_train"]["period_rmse"] = (
            self.__optuna_period_rmse.get(pair, -1)
        )

    def eval_set_and_weights(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame, test_weights: np.ndarray
    ) -> tuple[list[tuple] | None, list[np.ndarray] | None]:
        if self.data_split_parameters.get("test_size", TEST_SIZE) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(X_test, y_test)]
            eval_weights = [test_weights]

        return eval_set, eval_weights

    def min_max_pred(self, pred_df: pd.DataFrame) -> tuple[float, float]:
        temperature = self.freqai_info.get("predictions_temperature", 140.0)
        min_pred = smoothed_min(
            pred_df[EXTREMA_COLUMN],
            temperature=temperature,
        )
        max_pred = smoothed_max(
            pred_df[EXTREMA_COLUMN],
            temperature=temperature,
        )
        return min_pred, max_pred

    def optuna_optimize(
        self,
        pair: str,
        namespace: str,
        objective: Callable[[optuna.Trial], float],
        params_storage: dict[str, dict],
        rmse_storage: dict[str, float],
    ) -> None:
        identifier = self.freqai_info.get("identifier")
        study = self.optuna_create_study(pair, f"{identifier}-{namespace}-{pair}")
        if not study:
            return

        if self.__optuna_config.get("warm_start"):
            self.optuna_enqueue_previous_best_params(pair, namespace, study)

        logger.info(f"Optuna {namespace} hyperopt started")
        start_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=self.__optuna_config.get("n_trials"),
                n_jobs=self.__optuna_config.get("n_jobs"),
                timeout=self.__optuna_config.get("timeout"),
                gc_after_trial=True,
            )
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"Optuna {namespace} hyperopt failed ({time_spent:.2f} secs): {str(e)}",
                exc_info=True,
            )
            return

        time_spent = time.time() - start_time
        if QuickAdapterRegressorV3.optuna_study_has_best_params(study):
            params_storage[pair] = study.best_params
            rmse_storage[pair] = study.best_value
            logger.info(f"Optuna {namespace} hyperopt done ({time_spent:.2f} secs)")
            for key, value in {
                "rmse": rmse_storage[pair],
                **params_storage[pair],
            }.items():
                logger.info(f"Optuna {namespace} hyperopt | {key:>20s} : {value}")
            self.optuna_save_best_params(pair, namespace, params_storage[pair])
        else:
            logger.error(
                f"Optuna {namespace} hyperopt failed ({time_spent:.2f} secs): no study best params found"
            )

    def optuna_storage(self, pair: str) -> optuna.storages.BaseStorage:
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self.__optuna_config.get("storage")
        if storage_backend == "sqlite":
            storage = f"sqlite:///{storage_dir}/{storage_filename}.sqlite"
        elif storage_backend == "file":
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{storage_dir}/{storage_filename}.log"
                )
            )
        else:
            raise ValueError(
                f"Unsupported optuna storage backend: {storage_backend}. Supported backends are 'sqlite' and 'file'."
            )
        return storage

    def optuna_create_study(
        self, pair: str, study_name: str
    ) -> Optional[optuna.study.Study]:
        try:
            storage = self.optuna_storage(pair)
        except Exception as e:
            logger.error(
                f"Failed to create optuna storage for {study_name}: {str(e)}",
                exc_info=True,
            )
            return None

        if self.__optuna_config.get("continuous"):
            QuickAdapterRegressorV3.optuna_study_delete(study_name, storage)

        try:
            return optuna.create_study(
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(
                    multivariate=True, group=True, seed=1
                ),
                pruner=optuna.pruners.HyperbandPruner(),
                direction=optuna.study.StudyDirection.MINIMIZE,
                storage=storage,
                load_if_exists=not self.__optuna_config.get("continuous"),
            )
        except Exception as e:
            logger.error(
                f"Failed to create optuna study {study_name}: {str(e)}", exc_info=True
            )
            return None

    def optuna_enqueue_previous_best_params(
        self, pair: str, namespace: str, study: optuna.study.Study
    ) -> None:
        if namespace == "hp":
            best_params = self.__optuna_hp_params.get(pair)
        elif namespace == "period":
            best_params = self.__optuna_period_params.get(pair)
        if best_params:
            study.enqueue_trial(best_params)
        else:
            best_params = self.optuna_load_best_params(pair, namespace)
            if best_params:
                study.enqueue_trial(best_params)

    def optuna_save_best_params(
        self, pair: str, namespace: str, best_params: dict
    ) -> None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        try:
            with best_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(best_params, write_file, indent=4)
        except Exception as e:
            logger.error(
                f"Failed to save optuna {namespace} best params for {pair}: {str(e)}",
                exc_info=True,
            )
            raise

    def optuna_load_best_params(self, pair: str, namespace: str) -> Optional[dict]:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None

    @staticmethod
    def optuna_study_delete(
        study_name: str, storage: optuna.storages.BaseStorage
    ) -> None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass

    @staticmethod
    def optuna_study_load(
        study_name: str, storage: optuna.storages.BaseStorage
    ) -> Optional[optuna.study.Study]:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            study = None
        return study

    @staticmethod
    def optuna_study_has_best_params(study: Optional[optuna.study.Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_params
            return True
        # file backend storage raises KeyError
        except KeyError:
            return False
        # sqlite backend storage raises ValueError
        except ValueError:
            return False


def get_callbacks(trial: optuna.Trial, regressor: str) -> list[Callable]:
    if regressor == "xgboost":
        callbacks = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ]
    elif regressor == "lightgbm":
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    else:
        raise ValueError(f"Unsupported regressor model: {regressor}")
    return callbacks


def train_regressor(
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    eval_set: Optional[list[tuple]],
    eval_weights: Optional[list[np.ndarray]],
    model_training_parameters: dict,
    init_model: Any = None,
    callbacks: list[Callable] = None,
) -> Any:
    if regressor == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            callbacks=callbacks,
            **model_training_parameters,
        )
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=init_model,
        )
    elif regressor == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(objective="regression", **model_training_parameters)
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            eval_metric="rmse",
            init_model=init_model,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Unsupported regressor model: {regressor}")
    return model


def round_to_nearest(value: float, step: int) -> int:
    """
    Round a value to the nearest multiple of a given step.
    :param value: The value to round.
    :param step: The step size to round to (must be non-zero).
    :return: The rounded value.
    :raises ValueError: If step is zero.
    """
    if step == 0:
        raise ValueError("step must be non-zero")
    return int(round(value / step) * step)


def period_objective(
    trial: optuna.Trial,
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: np.ndarray,
    fit_live_predictions_candles: int,
    label_period_candles: int,
    candles_step: int,
    model_training_parameters: dict,
) -> float:
    min_train_window: int = fit_live_predictions_candles * 2
    max_train_window: int = len(X)
    if max_train_window < min_train_window:
        min_train_window = max_train_window
    train_window: int = trial.suggest_int(
        "train_period_candles", min_train_window, max_train_window, step=candles_step
    )
    X = X.iloc[-train_window:]
    y = y.iloc[-train_window:]
    train_weights = train_weights[-train_window:]

    min_test_window: int = fit_live_predictions_candles
    max_test_window: int = len(X_test)
    if max_test_window < min_test_window:
        min_test_window = max_test_window
    test_window: int = trial.suggest_int(
        "test_period_candles", min_test_window, max_test_window, step=candles_step
    )
    X_test = X_test.iloc[-test_window:]
    y_test = y_test.iloc[-test_window:]
    test_weights = test_weights[-test_window:]

    model = train_regressor(
        regressor=regressor,
        X=X,
        y=y,
        train_weights=train_weights,
        eval_set=[(X_test, y_test)],
        eval_weights=[test_weights],
        model_training_parameters=model_training_parameters,
        callbacks=get_callbacks(trial, regressor),
    )
    y_pred = model.predict(X_test)

    # TODO: implement a label_period_candles optimization compatible with ZigZag
    label_period_candles: int = trial.suggest_int(
        "label_period_candles",
        label_period_candles,
        label_period_candles,
        step=candles_step,
    )

    # min_label_period_candles: int = round_to_nearest(
    #     max(fit_live_predictions_candles // 12, 20), candles_step
    # )
    # max_label_period_candles: int = round_to_nearest(
    #     max(fit_live_predictions_candles // 4, min_label_period_candles),
    #     candles_step,
    # )
    # label_period_candles: int = trial.suggest_int(
    #     "label_period_candles",
    #     min_label_period_candles,
    #     max_label_period_candles,
    #     step=candles_step,
    # )
    # if label_period_candles > test_window:
    #     return float("inf")
    # label_periods_candles: int = (
    #     test_window // label_period_candles
    # ) * label_period_candles
    # if label_periods_candles == 0:
    #     return float("inf")
    # y_test = y_test.iloc[-label_periods_candles:]
    # test_weights = test_weights[-label_periods_candles:]
    # y_pred = y_pred[-label_periods_candles:]

    error = sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )

    return error


def get_optuna_study_model_parameters(trial: optuna.Trial, regressor: str) -> dict:
    study_model_parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if regressor == "xgboost":
        study_model_parameters.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            }
        )
    elif regressor == "lightgbm":
        study_model_parameters.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", 1e-8, 1.0, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            }
        )
    return study_model_parameters


def hp_objective(
    trial: optuna.Trial,
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: np.ndarray,
    model_training_parameters: dict,
) -> float:
    study_model_parameters = get_optuna_study_model_parameters(trial, regressor)
    model_training_parameters = {**model_training_parameters, **study_model_parameters}

    model = train_regressor(
        regressor=regressor,
        X=X,
        y=y,
        train_weights=train_weights,
        eval_set=[(X_test, y_test)],
        eval_weights=[test_weights],
        model_training_parameters=model_training_parameters,
        callbacks=get_callbacks(trial, regressor),
    )
    y_pred = model.predict(X_test)

    error = sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )

    return error


def smoothed_max(series: pd.Series, temperature=1.0) -> float:
    return sp.special.logsumexp(temperature * series.to_numpy()) / temperature


def smoothed_min(series: pd.Series, temperature=1.0) -> float:
    return -sp.special.logsumexp(-temperature * series.to_numpy()) / temperature
