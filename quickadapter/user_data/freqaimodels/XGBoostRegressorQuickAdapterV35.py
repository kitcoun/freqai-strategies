import logging
import json
from typing import Any
from pathlib import Path

from xgboost import XGBRegressor
import time
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
import pandas as pd
import scipy as sp
import optuna
import sklearn
import warnings

N_TRIALS = 36
TEST_SIZE = 0.1

EXTREMA_COLUMN = "&s-extrema"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class XGBoostRegressorQuickAdapterV35(BaseRegressionModel):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pairs = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI model requires StaticPairList method defined in pairlists configuration and pair_whitelist defined in exchange section configuration"
            )
        self.__optuna_config = self.freqai_info.get("optuna_hyperopt", {})
        self.__optuna_hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.__optuna_config.get("enabled", False)
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

        xgb_model = self.get_init_model(dk.pair)

        start = time.time()
        if self.__optuna_hyperopt:
            optuna_hp_params, optuna_hp_rmse = self.optuna_hp_optimize(
                dk.pair, X, y, train_weights, X_test, y_test, test_weights
            )
            if optuna_hp_params:
                self.__optuna_hp_params[dk.pair] = optuna_hp_params
            if optuna_hp_rmse:
                self.__optuna_hp_rmse[dk.pair] = optuna_hp_rmse

            if self.__optuna_hp_params.get(dk.pair):
                model_training_parameters = {
                    **model_training_parameters,
                    **self.__optuna_hp_params[dk.pair],
                }

            optuna_period_params, optuna_period_rmse = self.optuna_period_optimize(
                dk.pair,
                X,
                y,
                train_weights,
                X_test,
                y_test,
                test_weights,
                model_training_parameters,
            )
            if optuna_period_params:
                self.__optuna_period_params[dk.pair] = optuna_period_params
            if optuna_period_rmse:
                self.__optuna_period_rmse[dk.pair] = optuna_period_rmse

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

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **model_training_parameters,
        )

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model,
        )
        time_spent = time.time() - start
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def get_label_period_candles(self, pair: str) -> int:
        if self.__optuna_period_params.get(pair, {}).get("label_period_candles"):
            return self.__optuna_period_params[pair]["label_period_candles"]
        return self.ft_params["label_period_candles"]

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
                    f"{pair}: fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go."
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
            label_period_candles = self.get_label_period_candles(pair)
            min_pred, max_pred = self.min_max_pred(
                pred_df_full,
                num_candles,
                label_period_candles,
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
            self.get_label_period_candles(pair)
        )
        dk.data["extra_returns_per_train"]["hp_rmse"] = self.__optuna_hp_rmse.get(
            pair, -1
        )
        dk.data["extra_returns_per_train"]["period_rmse"] = (
            self.__optuna_period_rmse.get(pair, -1)
        )

    def eval_set_and_weights(self, X_test, y_test, test_weights):
        if self.data_split_parameters.get("test_size", TEST_SIZE) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(X_test, y_test)]
            eval_weights = [test_weights]

        return eval_set, eval_weights

    def optuna_storage(self, pair: str) -> optuna.storages.BaseStorage | None:
        storage_dir = str(self.full_path)
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self.__optuna_config.get("storage", "file")
        if storage_backend == "sqlite":
            storage = f"sqlite:///{storage_dir}/{storage_filename}.sqlite"
        elif storage_backend == "file":
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{storage_dir}/{storage_filename}.log"
                )
            )
        return storage

    def min_max_pred(
        self,
        pred_df: pd.DataFrame,
        fit_live_predictions_candles: int,
        label_period_candles: int,
    ) -> tuple[pd.Series, pd.Series]:
        prediction_thresholds_smoothing = self.freqai_info.get(
            "prediction_thresholds_smoothing", "mean"
        )
        smoothing_methods: dict = {
            "quantile": self.quantile_min_max_pred,
            "mean": mean_min_max_pred,
            "median": median_min_max_pred,
        }
        return smoothing_methods.get(
            prediction_thresholds_smoothing, smoothing_methods["mean"]
        )(pred_df, fit_live_predictions_candles, label_period_candles)

    def optuna_hp_enqueue_previous_best_trial(
        self,
        pair: str,
        study: optuna.study.Study,
    ) -> None:
        study_namespace = "hp"
        if self.__optuna_hp_params.get(pair):
            study.enqueue_trial(self.__optuna_hp_params[pair])
        elif self.optuna_load_best_params(pair, study_namespace):
            study.enqueue_trial(self.optuna_load_best_params(pair, study_namespace))

    def optuna_hp_optimize(
        self,
        pair: str,
        X,
        y,
        train_weights,
        X_test,
        y_test,
        test_weights,
    ) -> tuple[dict, float] | tuple[None, None]:
        identifier = self.freqai_info.get("identifier")
        study_namespace = "hp"
        study_name = f"{identifier}-{study_namespace}-{pair}"
        storage = self.optuna_storage(pair)
        pruner = optuna.pruners.HyperbandPruner()
        if self.__optuna_config.get("continuous", True):
            self.optuna_study_delete(study_name, storage)
        study = optuna.create_study(
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
            ),
            pruner=pruner,
            direction=optuna.study.StudyDirection.MINIMIZE,
            storage=storage,
            load_if_exists=not self.__optuna_config.get("continuous", True),
        )
        if self.__optuna_config.get("warm_start", True):
            self.optuna_hp_enqueue_previous_best_trial(pair, study)
        logger.info(f"Optuna {study_namespace} hyperopt started")
        start = time.time()
        try:
            study.optimize(
                lambda trial: hp_objective(
                    trial,
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.model_training_parameters,
                ),
                n_trials=self.__optuna_config.get("n_trials", N_TRIALS),
                n_jobs=min(
                    self.__optuna_config.get("n_jobs", 1),
                    max(int(self.max_system_threads / 4), 1),
                ),
                timeout=self.__optuna_config.get("timeout", 3600),
                gc_after_trial=True,
            )
        except Exception as e:
            logger.error(
                f"Optuna {study_namespace} hyperopt failed: {e}", exc_info=True
            )
            return None, None
        time_spent = time.time() - start
        logger.info(f"Optuna {study_namespace} hyperopt done ({time_spent:.2f} secs)")

        params = study.best_params
        self.optuna_save_best_params(pair, study_namespace, params)
        # log params
        for key, value in {"rmse": study.best_value, **params}.items():
            logger.info(f"Optuna {study_namespace} hyperopt | {key:>20s} : {value}")
        return params, study.best_value

    def optuna_period_enqueue_previous_best_trial(
        self,
        pair: str,
        study: optuna.study.Study,
    ) -> None:
        study_namespace = "period"
        if self.__optuna_period_params.get(pair):
            study.enqueue_trial(self.__optuna_period_params[pair])
        elif self.optuna_load_best_params(pair, study_namespace):
            study.enqueue_trial(self.optuna_load_best_params(pair, study_namespace))

    def optuna_period_optimize(
        self,
        pair: str,
        X,
        y,
        train_weights,
        X_test,
        y_test,
        test_weights,
        model_training_parameters,
    ) -> tuple[dict, float] | tuple[None, None]:
        identifier = self.freqai_info.get("identifier")
        study_namespace = "period"
        study_name = f"{identifier}-{study_namespace}-{pair}"
        storage = self.optuna_storage(pair)
        pruner = optuna.pruners.HyperbandPruner()
        if self.__optuna_config.get("continuous", True):
            self.optuna_study_delete(study_name, storage)
        study = optuna.create_study(
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
            ),
            pruner=pruner,
            direction=optuna.study.StudyDirection.MINIMIZE,
            storage=storage,
            load_if_exists=not self.__optuna_config.get("continuous", True),
        )
        if self.__optuna_config.get("warm_start", True):
            self.optuna_period_enqueue_previous_best_trial(pair, study)
        logger.info(f"Optuna {study_namespace} hyperopt started")
        start = time.time()
        try:
            study.optimize(
                lambda trial: period_objective(
                    trial,
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.data_split_parameters.get("test_size", TEST_SIZE),
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self.__optuna_config.get("candles_step", 10),
                    model_training_parameters,
                ),
                n_trials=self.__optuna_config.get("n_trials", N_TRIALS),
                n_jobs=min(
                    self.__optuna_config.get("n_jobs", 1),
                    max(int(self.max_system_threads / 4), 1),
                ),
                timeout=self.__optuna_config.get("timeout", 3600),
                gc_after_trial=True,
            )
        except Exception as e:
            logger.error(
                f"Optuna {study_namespace} hyperopt failed: {e}", exc_info=True
            )
            return None, None
        time_spent = time.time() - start
        logger.info(f"Optuna {study_namespace} hyperopt done ({time_spent:.2f} secs)")

        params = study.best_params
        self.optuna_save_best_params(pair, study_namespace, params)
        # log params
        for key, value in {"rmse": study.best_value, **params}.items():
            logger.info(f"Optuna {study_namespace} hyperopt | {key:>20s} : {value}")
        return params, study.best_value

    def optuna_save_best_params(
        self, pair: str, namespace: str, best_params: dict
    ) -> None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        with best_params_path.open("w", encoding="utf-8") as write_file:
            json.dump(best_params, write_file, indent=4)

    def optuna_load_best_params(self, pair: str, namespace: str) -> dict | None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None

    def optuna_study_delete(
        self, study_name: str, storage: optuna.storages.BaseStorage
    ) -> None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass

    def optuna_study_load(
        self, study_name: str, storage: optuna.storages.BaseStorage
    ) -> optuna.study.Study | None:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            study = None
        return study

    def optuna_study_has_best_params(self, study: optuna.study.Study | None) -> bool:
        if not study:
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

    def quantile_min_max_pred(
        self,
        pred_df: pd.DataFrame,
        fit_live_predictions_candles: int,
        label_period_candles: int,
    ) -> tuple[pd.Series, pd.Series]:
        pred_df_sorted = (
            pred_df.select_dtypes(exclude=["object"])
            .copy()
            .apply(lambda col: col.sort_values(ascending=False, ignore_index=True))
        )

        label_period_frequency: int = int(
            fit_live_predictions_candles / (label_period_candles * 2)
        )
        q = self.freqai_info.get("quantile", 0.75)
        min_pred = pred_df_sorted.iloc[-label_period_frequency:].quantile(1 - q)
        max_pred = pred_df_sorted.iloc[:label_period_frequency].quantile(q)
        return min_pred[EXTREMA_COLUMN], max_pred[EXTREMA_COLUMN]


def mean_min_max_pred(
    pred_df: pd.DataFrame, fit_live_predictions_candles: int, label_period_candles: int
) -> tuple[pd.Series, pd.Series]:
    pred_df_sorted = (
        pred_df.select_dtypes(exclude=["object"])
        .copy()
        .apply(lambda col: col.sort_values(ascending=False, ignore_index=True))
    )

    label_period_frequency: int = int(
        fit_live_predictions_candles / (label_period_candles * 2)
    )
    min_pred = pred_df_sorted.iloc[-label_period_frequency:].mean()
    max_pred = pred_df_sorted.iloc[:label_period_frequency].mean()
    return min_pred[EXTREMA_COLUMN], max_pred[EXTREMA_COLUMN]


def median_min_max_pred(
    pred_df: pd.DataFrame, fit_live_predictions_candles: int, label_period_candles: int
) -> tuple[pd.Series, pd.Series]:
    pred_df_sorted = (
        pred_df.select_dtypes(exclude=["object"])
        .copy()
        .apply(lambda col: col.sort_values(ascending=False, ignore_index=True))
    )

    label_period_frequency: int = int(
        fit_live_predictions_candles / (label_period_candles * 2)
    )
    min_pred = pred_df_sorted.iloc[-label_period_frequency:].median()
    max_pred = pred_df_sorted.iloc[:label_period_frequency].median()
    return min_pred[EXTREMA_COLUMN], max_pred[EXTREMA_COLUMN]


def period_objective(
    trial,
    X,
    y,
    train_weights,
    X_test,
    y_test,
    test_weights,
    test_size,
    fit_live_predictions_candles: int,
    candles_step: int,
    model_training_parameters,
) -> float:
    min_train_window: int = fit_live_predictions_candles * 2
    max_train_window: int = max(len(X), min_train_window)
    train_window: int = trial.suggest_int(
        "train_period_candles", min_train_window, max_train_window, step=candles_step
    )
    X = X.iloc[-train_window:]
    y = y.iloc[-train_window:]
    train_weights = train_weights[-train_window:]

    min_test_window: int = max(
        int(min_train_window * test_size), fit_live_predictions_candles
    )
    max_test_window: int = max(len(X_test), min_test_window)
    test_window: int = trial.suggest_int(
        "test_period_candles", min_test_window, max_test_window, step=candles_step
    )
    X_test = X_test.iloc[-test_window:]
    y_test = y_test.iloc[-test_window:]
    test_weights = test_weights[-test_window:]

    # Fit the model
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ],
        **model_training_parameters,
    )
    model.fit(
        X=X,
        y=y,
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[test_weights],
    )
    y_pred = model.predict(X_test)

    min_label_period_candles: int = fit_live_predictions_candles // 10
    max_label_period_candles: int = max(
        fit_live_predictions_candles // 6, min_label_period_candles
    )
    label_period_candles: int = trial.suggest_int(
        "label_period_candles",
        min_label_period_candles,
        max_label_period_candles,
        step=candles_step,
    )
    label_window: int = label_period_candles * 2
    label_windows_length: int = (
        fit_live_predictions_candles // label_window
    ) * label_window
    y_test = y_test.iloc[-label_windows_length:].to_numpy()
    test_weights = test_weights[-label_windows_length:]
    y_pred = y_pred[-label_windows_length:]
    y_test = [y_test[i : i + label_window] for i in range(0, len(y_test), label_window)]
    test_weights = [
        test_weights[i : i + label_window]
        for i in range(0, len(test_weights), label_window)
    ]
    y_pred = [y_pred[i : i + label_window] for i in range(0, len(y_pred), label_window)]

    error = 0.0
    for y_t, y_p, t_w in zip(y_test, y_pred, test_weights):
        error += sklearn.metrics.root_mean_squared_error(y_t, y_p, sample_weight=t_w)
    error /= len(y_test)

    return error


def hp_objective(
    trial, X, y, train_weights, X_test, y_test, test_weights, model_training_parameters
) -> float:
    study_parameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 18),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model_training_parameters = {**model_training_parameters, **study_parameters}

    # Fit the model
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        callbacks=[
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ],
        **model_training_parameters,
    )
    model.fit(
        X=X,
        y=y,
        sample_weight=train_weights,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[test_weights],
    )
    y_pred = model.predict(X_test)

    error = sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )

    return error
