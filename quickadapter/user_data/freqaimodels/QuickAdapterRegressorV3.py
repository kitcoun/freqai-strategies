from enum import IntEnum
import logging
import json
import time
import numpy as np
import pandas as pd
import scipy as sp
import optuna
import sklearn
import warnings
import talib.abstract as ta

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

    version = "3.7.29"

    @cached_property
    def _optuna_config(self) -> dict:
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
            "seed": 1,
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
        self._optuna_hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self._optuna_config.get("enabled")
            and self.data_split_parameters.get("test_size", TEST_SIZE) > 0
        )
        self._optuna_hp_value: dict[str, float] = {}
        self._optuna_train_value: dict[str, float] = {}
        self._optuna_label_values: dict[str, dict] = {}
        self._optuna_hp_params: dict[str, dict] = {}
        self._optuna_train_params: dict[str, dict] = {}
        self._optuna_label_params: dict[str, dict] = {}
        for pair in self.pairs:
            self._optuna_hp_value[pair] = -1
            self._optuna_train_value[pair] = -1
            self._optuna_label_values[pair] = [-1, -1]
            self._optuna_hp_params[pair] = (
                self.optuna_load_best_params(pair, "hp")
                if self.optuna_load_best_params(pair, "hp")
                else {}
            )
            self._optuna_train_params[pair] = (
                self.optuna_load_best_params(pair, "train")
                if self.optuna_load_best_params(pair, "train")
                else {}
            )
            self._optuna_label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": self.ft_params.get(
                        "label_period_candles", 50
                    ),
                    "label_natr_ratio": self.ft_params.get("label_natr_ratio", 0.12125),
                }
            )
        logger.info(
            f"Initialized {self.__class__.__name__} {self.freqai_info.get('regressor', 'xgboost')} regressor model version {self.version}"
        )

    def get_optuna_params(self, pair: str, namespace: str) -> dict[str, Any]:
        if namespace == "hp":
            params = self._optuna_hp_params.get(pair)
        elif namespace == "train":
            params = self._optuna_train_params.get(pair)
        elif namespace == "label":
            params = self._optuna_label_params.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return params

    def set_optuna_params(self, pair: str, namespace: str, params: dict) -> None:
        if namespace == "hp":
            self._optuna_hp_params[pair] = params
        elif namespace == "train":
            self._optuna_train_params[pair] = params
        elif namespace == "label":
            self._optuna_label_params[pair] = params
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def get_optuna_value(self, pair: str, namespace: str) -> float:
        if namespace == "hp":
            value = self._optuna_hp_value.get(pair)
        elif namespace == "train":
            value = self._optuna_train_value.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return value

    def set_optuna_value(self, pair: str, namespace: str, value: float) -> None:
        if namespace == "hp":
            self._optuna_hp_value[pair] = value
        elif namespace == "train":
            self._optuna_train_value[pair] = value
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def get_optuna_values(self, pair: str, namespace: str) -> list:
        if namespace == "label":
            values = self._optuna_label_values.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return values

    def set_optuna_values(self, pair: str, namespace: str, values: list) -> None:
        if namespace == "label":
            self._optuna_label_values[pair] = values
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        :param dk: the FreqaiDataKitchen object
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        train_weights = data_dictionary["train_weights"]

        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"]
        test_weights = data_dictionary["test_weights"]

        model_training_parameters = self.model_training_parameters

        start = time.time()
        if self._optuna_hyperopt:
            self.optuna_optimize(
                pair=dk.pair,
                namespace="label",
                objective=lambda trial: label_objective(
                    trial,
                    self.data_provider.get_pair_dataframe(dk.pair),
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self._optuna_config.get("candles_step"),
                ),
                directions=[
                    optuna.study.StudyDirection.MAXIMIZE,
                    optuna.study.StudyDirection.MAXIMIZE,
                ],
            )

            self.optuna_optimize(
                pair=dk.pair,
                namespace="hp",
                objective=lambda trial: hp_objective(
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
                direction=optuna.study.StudyDirection.MINIMIZE,
            )

            optuna_hp_params = self.get_optuna_params(dk.pair, "hp")
            if optuna_hp_params:
                model_training_parameters = {
                    **model_training_parameters,
                    **optuna_hp_params,
                }

            self.optuna_optimize(
                pair=dk.pair,
                namespace="train",
                objective=lambda trial: train_objective(
                    trial,
                    self.freqai_info.get("regressor", "xgboost"),
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.data_split_parameters.get("test_size", TEST_SIZE),
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self._optuna_config.get("candles_step"),
                    model_training_parameters,
                ),
                direction=optuna.study.StudyDirection.MINIMIZE,
            )

            optuna_train_params = self.get_optuna_params(dk.pair, "train")
            if optuna_train_params:
                train_window = optuna_train_params.get("train_period_candles")
                X = X.iloc[-train_window:]
                y = y.iloc[-train_window:]
                train_weights = train_weights[-train_window:]

                test_window = optuna_train_params.get("test_period_candles")
                X_test = X_test.iloc[-test_window:]
                y_test = y_test.iloc[-test_window:]
                test_weights = test_weights[-test_window:]

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        model = fit_regressor(
            regressor=self.freqai_info.get("regressor", "xgboost"),
            X=X,
            y=y,
            train_weights=train_weights,
            eval_set=eval_set,
            eval_weights=eval_weights,
            model_training_parameters=model_training_parameters,
            init_model=self.get_init_model(dk.pair),
        )
        time_spent = time.time() - start
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        warmed_up = True

        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles", 100
        )

        if self.live:
            if not hasattr(self, "exchange_candles"):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candles_diff = len(self.dd.historic_predictions[pair].index) - (
                fit_live_predictions_candles + self.exchange_candles
            )
            if candles_diff < 0:
                logger.warning(
                    f"{pair}: fit live predictions not warmed up yet. Still {abs(candles_diff)} candles to go."
                )
                warmed_up = False

        pred_df_full = (
            self.dd.historic_predictions[pair]
            .iloc[-fit_live_predictions_candles:]
            .reset_index(drop=True)
        )

        if not warmed_up:
            dk.data["extra_returns_per_train"][MINIMA_THRESHOLD_COLUMN] = -2
            dk.data["extra_returns_per_train"][MAXIMA_THRESHOLD_COLUMN] = 2
        else:
            min_pred, max_pred = self.min_max_pred(
                pred_df_full,
                fit_live_predictions_candles,
                self.get_optuna_params(pair, "label").get("label_period_candles"),
            )
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
            self.get_optuna_params(pair, "label").get("label_period_candles")
        )
        dk.data["extra_returns_per_train"]["label_natr_ratio"] = self.get_optuna_params(
            pair, "label"
        ).get("label_natr_ratio")

        dk.data["extra_returns_per_train"]["hp_rmse"] = self.get_optuna_value(
            pair, "hp"
        )
        dk.data["extra_returns_per_train"]["train_rmse"] = self.get_optuna_value(
            pair, "train"
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

    def min_max_pred(
        self,
        pred_df: pd.DataFrame,
        fit_live_predictions_candles: int,
        label_period_candles: int,
    ) -> tuple[float, float]:
        temperature = float(
            self.freqai_info.get("prediction_thresholds_temperature", 250.0)
        )
        extrema = pred_df[EXTREMA_COLUMN].iloc[
            -(
                max(2, int(fit_live_predictions_candles / label_period_candles))
                * label_period_candles
            ) :
        ]
        min_pred = smoothed_min(extrema, temperature=temperature)
        max_pred = smoothed_max(extrema, temperature=temperature)
        return min_pred, max_pred

    def get_multi_objective_study_best_trial(
        self, namespace: str, study: optuna.study.Study
    ) -> Optional[optuna.trial.FrozenTrial]:
        if not QuickAdapterRegressorV3.optuna_study_has_best_trials(study):
            return None
        best_trials = study.best_trials
        if namespace == "label":
            pivots_sizes = [trial.values[1] for trial in best_trials]
            quantile_pivots_size = np.quantile(
                pivots_sizes, self.ft_params.get("label_quantile", 0.75)
            )
            equal_quantile_pivots_size_trials = [
                trial
                for trial in best_trials
                if np.isclose(trial.values[1], quantile_pivots_size)
            ]
            if equal_quantile_pivots_size_trials:
                return max(
                    equal_quantile_pivots_size_trials, key=lambda trial: trial.values[0]
                )
            nearest_above_quantile = (
                np.inf,
                -np.inf,
                None,
            )  # (trial_pivots_size, trial_scaled_natr, trial_index)
            nearest_below_quantile = (
                -np.inf,
                -np.inf,
                None,
            )  # (trial_pivots_size, trial_scaled_natr, trial_index)
            for idx, trial in enumerate(best_trials):
                pivots_size = trial.values[1]
                if pivots_size >= quantile_pivots_size:
                    if pivots_size < nearest_above_quantile[0] or (
                        pivots_size == nearest_above_quantile[0]
                        and trial.values[0] > nearest_above_quantile[1]
                    ):
                        nearest_above_quantile = (pivots_size, trial.values[0], idx)
                if pivots_size <= quantile_pivots_size:
                    if pivots_size > nearest_below_quantile[0] or (
                        pivots_size == nearest_below_quantile[0]
                        and trial.values[0] > nearest_below_quantile[1]
                    ):
                        nearest_below_quantile = (pivots_size, trial.values[0], idx)
            if nearest_above_quantile[2] is None or nearest_below_quantile[2] is None:
                return None
            above_quantile_trial = best_trials[nearest_above_quantile[2]]
            below_quantile_trial = best_trials[nearest_below_quantile[2]]
            if above_quantile_trial.values[0] >= below_quantile_trial.values[0]:
                return above_quantile_trial
            else:
                return below_quantile_trial
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def optuna_optimize(
        self,
        pair: str,
        namespace: str,
        objective: Callable[[optuna.trial.Trial], float],
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> None:
        identifier = self.freqai_info.get("identifier")
        study = self.optuna_create_study(
            pair=pair,
            study_name=f"{identifier}-{pair}-{namespace}",
            direction=direction,
            directions=directions,
        )
        if not study:
            return

        if self._optuna_config.get("warm_start"):
            self.optuna_enqueue_previous_best_params(pair, namespace, study)

        is_study_single_objective = direction is not None and directions is None
        if is_study_single_objective is True:
            objective_type = "single objective"
        else:
            objective_type = "multi objective"
        logger.info(f"Optuna {pair} {namespace} {objective_type} hyperopt started")
        start_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=self._optuna_config.get("n_trials"),
                n_jobs=self._optuna_config.get("n_jobs"),
                timeout=self._optuna_config.get("timeout"),
                gc_after_trial=True,
            )
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"Optuna {pair} {namespace} {objective_type} hyperopt failed ({time_spent:.2f} secs): {str(e)}",
                exc_info=True,
            )
            return

        time_spent = time.time() - start_time
        if is_study_single_objective:
            if not QuickAdapterRegressorV3.optuna_study_has_best_trial(study):
                logger.error(
                    f"Optuna {pair} {namespace} {objective_type} hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_value(pair, namespace, study.best_value)
            self.set_optuna_params(pair, namespace, study.best_params)
            study_results = {
                "value": self.get_optuna_value(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
        else:
            best_trial = self.get_multi_objective_study_best_trial("label", study)
            if not best_trial:
                logger.error(
                    f"Optuna {pair} {namespace} {objective_type} hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_values(pair, namespace, best_trial.values)
            self.set_optuna_params(pair, namespace, best_trial.params)
            study_results = {
                "values": self.get_optuna_values(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
        logger.info(
            f"Optuna {pair} {namespace} {objective_type} done ({time_spent:.2f} secs)"
        )
        for key, value in study_results.items():
            logger.info(
                f"Optuna {pair} {namespace} {objective_type} hyperopt | {key:>20s} : {value}"
            )
        self.optuna_save_best_params(pair, namespace)

    def optuna_storage(self, pair: str) -> optuna.storages.BaseStorage:
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self._optuna_config.get("storage")
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
        self,
        pair: str,
        study_name: str,
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> Optional[optuna.study.Study]:
        try:
            storage = self.optuna_storage(pair)
        except Exception as e:
            logger.error(
                f"Failed to create optuna storage for {study_name}: {str(e)}",
                exc_info=True,
            )
            return None

        if self._optuna_config.get("continuous"):
            QuickAdapterRegressorV3.optuna_study_delete(study_name, storage)

        try:
            return optuna.create_study(
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(
                    multivariate=True, group=True, seed=self._optuna_config.get("seed")
                ),
                pruner=optuna.pruners.HyperbandPruner(),
                direction=direction,
                directions=directions,
                storage=storage,
                load_if_exists=not self._optuna_config.get("continuous"),
            )
        except Exception as e:
            logger.error(
                f"Failed to create optuna study {study_name}: {str(e)}", exc_info=True
            )
            return None

    def optuna_enqueue_previous_best_params(
        self, pair: str, namespace: str, study: optuna.study.Study
    ) -> None:
        best_params = self.get_optuna_params(pair, namespace)
        if best_params:
            study.enqueue_trial(best_params)
        else:
            best_params = self.optuna_load_best_params(pair, namespace)
            if best_params:
                study.enqueue_trial(best_params)

    def optuna_save_best_params(self, pair: str, namespace: str) -> None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        try:
            with best_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(self.get_optuna_params(pair, namespace), write_file, indent=4)
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
    def optuna_study_has_best_trial(study: Optional[optuna.study.Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trial
            return True
        # file backend storage raises KeyError
        except KeyError:
            return False
        # sqlite backend storage raises ValueError
        except ValueError:
            return False

    @staticmethod
    def optuna_study_has_best_trials(study: Optional[optuna.study.Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trials
            return True
        # file backend storage raises KeyError
        except KeyError:
            return False
        # sqlite backend storage raises ValueError
        except ValueError:
            return False


def get_callbacks(trial: optuna.trial.Trial, regressor: str) -> list[Callable]:
    if regressor == "xgboost":
        callbacks = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ]
    elif regressor == "lightgbm":
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    else:
        raise ValueError(f"Unsupported regressor model: {regressor}")
    return callbacks


def fit_regressor(
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


def train_objective(
    trial: optuna.trial.Trial,
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: np.ndarray,
    test_size: float,
    fit_live_predictions_candles: int,
    candles_step: int,
    model_training_parameters: dict,
) -> float:
    min_train_window: int = fit_live_predictions_candles * int(1 / test_size)
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

    model = fit_regressor(
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


def get_optuna_study_model_parameters(
    trial: optuna.trial.Trial, regressor: str
) -> dict:
    study_model_parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
    }
    if regressor == "xgboost":
        study_model_parameters.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            }
        )
    elif regressor == "lightgbm":
        study_model_parameters.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 8, 256),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", 1e-8, 10.0, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            }
        )
    return study_model_parameters


def hp_objective(
    trial: optuna.trial.Trial,
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

    model = fit_regressor(
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


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_ratio: float = 1.0,
    confirmation_window: int = 2,
    depth: int = 12,
) -> tuple[list[int], list[float], list[int]]:
    if df.empty or len(df) < max(natr_period, 2 * confirmation_window + 1):
        return [], [], []

    indices = df.index.tolist()
    thresholds = (
        (ta.NATR(df, timeperiod=natr_period) * natr_ratio).fillna(method="bfill").values
    )
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    state: TrendDirection = TrendDirection.NEUTRAL

    last_pivot_pos = -depth - 1
    pivots_indices, pivots_values, pivots_directions = [], [], []

    candidate_pivot_pos = -1
    candidate_pivot_value = np.nan
    candidate_pivot_direction: TrendDirection = TrendDirection.NEUTRAL

    def update_candidate_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal candidate_pivot_pos, candidate_pivot_value, candidate_pivot_direction
        candidate_pivot_pos = pos
        candidate_pivot_value = value
        candidate_pivot_direction = direction

    def reset_candidate_pivot():
        nonlocal candidate_pivot_pos, candidate_pivot_value, candidate_pivot_direction
        candidate_pivot_pos = -1
        candidate_pivot_value = np.nan
        candidate_pivot_direction = TrendDirection.NEUTRAL

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)
        last_pivot_pos = pos

    def is_reversal_confirmed(pos: int, direction: TrendDirection) -> bool:
        if pos - confirmation_window < 0 or pos + confirmation_window >= len(df):
            return False
        next_slice = slice(pos + 1, pos + confirmation_window + 1)
        next_closes = closes[next_slice]
        next_highs = highs[next_slice]
        next_lows = lows[next_slice]
        previous_slice = slice(pos - confirmation_window, pos)
        previous_closes = closes[previous_slice]
        previous_highs = highs[previous_slice]
        previous_lows = lows[previous_slice]

        if direction == TrendDirection.DOWN:
            return (
                np.all(next_closes < highs[pos])
                and np.all(previous_closes < highs[pos])
                and np.max(next_highs) <= highs[pos]
                and np.max(previous_highs) <= highs[pos]
            )
        elif direction == TrendDirection.UP:
            return (
                np.all(next_closes > lows[pos])
                and np.all(previous_closes > lows[pos])
                and np.min(next_lows) >= lows[pos]
                and np.min(previous_lows) >= lows[pos]
            )
        return False

    start_pos = 0
    initial_high_pos = start_pos
    initial_low_pos = start_pos
    initial_high = highs[initial_high_pos]
    initial_low = lows[initial_low_pos]
    for i in range(start_pos + 1, len(df)):
        current_high = highs[i]
        current_low = lows[i]
        if current_high > initial_high:
            initial_high, initial_high_pos = current_high, i
        if current_low < initial_low:
            initial_low, initial_low_pos = current_low, i

        initial_move_from_high = (initial_high - current_low) / initial_high
        initial_move_from_low = (current_high - initial_low) / initial_low
        if initial_move_from_high >= thresholds[
            initial_high_pos
        ] and is_reversal_confirmed(initial_high_pos, TrendDirection.DOWN):
            add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
            state = TrendDirection.DOWN
            break
        elif initial_move_from_low >= thresholds[
            initial_low_pos
        ] and is_reversal_confirmed(initial_low_pos, TrendDirection.UP):
            add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
            state = TrendDirection.UP
            break
    else:
        return [], [], []

    for i in range(last_pivot_pos + 1, len(df)):
        current_high = highs[i]
        current_low = lows[i]

        if state == TrendDirection.UP:
            if np.isnan(candidate_pivot_value) or current_high > candidate_pivot_value:
                update_candidate_pivot(i, current_high, TrendDirection.UP)
            if (
                (candidate_pivot_value - current_low) / candidate_pivot_value
                >= thresholds[candidate_pivot_pos]
                and (candidate_pivot_pos - last_pivot_pos) >= depth
                and is_reversal_confirmed(candidate_pivot_pos, TrendDirection.DOWN)
            ):
                add_pivot(candidate_pivot_pos, candidate_pivot_value, TrendDirection.UP)
                reset_candidate_pivot()
                state = TrendDirection.DOWN
        elif state == TrendDirection.DOWN:
            if np.isnan(candidate_pivot_value) or current_low < candidate_pivot_value:
                update_candidate_pivot(i, current_low, TrendDirection.DOWN)
            if (
                (current_high - candidate_pivot_value) / candidate_pivot_value
                >= thresholds[candidate_pivot_pos]
                and (candidate_pivot_pos - last_pivot_pos) >= depth
                and is_reversal_confirmed(candidate_pivot_pos, TrendDirection.UP)
            ):
                add_pivot(
                    candidate_pivot_pos, candidate_pivot_value, TrendDirection.DOWN
                )
                reset_candidate_pivot()
                state = TrendDirection.UP

    return pivots_indices, pivots_values, pivots_directions


def label_objective(
    trial: optuna.trial.Trial,
    df: pd.DataFrame,
    fit_live_predictions_candles: int,
    candles_step: int,
) -> tuple[float, int]:
    min_label_period_candles: int = round_to_nearest_int(
        max(fit_live_predictions_candles // 16, 20), candles_step
    )
    max_label_period_candles: int = round_to_nearest_int(
        max(fit_live_predictions_candles // 4, min_label_period_candles),
        candles_step,
    )
    label_period_candles = trial.suggest_int(
        "label_period_candles",
        min_label_period_candles,
        max_label_period_candles,
        step=candles_step,
    )
    label_natr_ratio = trial.suggest_float("label_natr_ratio", 0.07, 0.2)

    df = df.iloc[
        -(
            max(2, int(fit_live_predictions_candles / label_period_candles))
            * label_period_candles
        ) :
    ]

    if df.empty:
        return -float("inf"), -float("inf")

    _, pivots_values, _ = zigzag(
        df,
        natr_period=label_period_candles,
        natr_ratio=label_natr_ratio,
    )

    if len(pivots_values) < 2:
        return -float("inf"), -float("inf")

    scaled_natr_label_period_candles = (
        ta.NATR(df, timeperiod=label_period_candles) * label_natr_ratio
    )

    return scaled_natr_label_period_candles.median(), len(pivots_values)


def smoothed_max(series: pd.Series, temperature=1.0) -> float:
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if np.isclose(temperature, 0):
        return data_array.max()
    return sp.special.logsumexp(temperature * data_array) / temperature


def smoothed_min(series: pd.Series, temperature=1.0) -> float:
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if np.isclose(temperature, 0):
        return data_array.min()
    return -sp.special.logsumexp(-temperature * data_array) / temperature


def boltzmann_operator(series: pd.Series, alpha: float) -> float:
    """
    Compute the Boltzmann operator of a series with parameter alpha.
    """
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if alpha == 0:
        return np.mean(data_array)
    scaled_data = alpha * data_array
    shifted_exponentials = np.exp(scaled_data - np.max(scaled_data))
    numerator = np.sum(data_array * shifted_exponentials)
    denominator = np.sum(shifted_exponentials)
    return numerator / denominator


def round_to_nearest_int(value: float, step: int) -> int:
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
