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

N_TRIALS = 36
TEST_SIZE = 0.25

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

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]

        X_test = data_dictionary["test_features"]
        y_test = data_dictionary["test_labels"]
        test_weights = data_dictionary["test_weights"]

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        sample_weight = data_dictionary["train_weights"]

        lgbm_model = self.get_init_model(dk.pair)
        start = time.time()
        hp = {}
        optuna_hyperopt: bool = (
            self.freqai_info.get("optuna_hyperopt", False)
            and self.freqai_info.get("data_split_parameters", {}).get(
                "test_size", TEST_SIZE
            )
            > 0
        )
        if optuna_hyperopt:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study = optuna.create_study(pruner=pruner, direction="minimize")
            study.optimize(
                lambda trial: objective(
                    trial,
                    X,
                    y,
                    sample_weight,
                    X_test,
                    y_test,
                    self.model_training_parameters,
                ),
                n_trials=N_TRIALS,
                n_jobs=1,
            )

            hp = study.best_params
            # trial = study.best_trial
            # log params
            for key, value in hp.items():
                logger.info(f"Optuna hyperopt {key:>20s} : {value}")
            logger.info(
                f"Optuna hyperopt {'best objective value':>20s} : {study.best_value}"
            )

        window = hp.get("train_period_candles", 4032)
        X = X.tail(window)
        y = y.tail(window)
        sample_weight = sample_weight[-window:]
        if optuna_hyperopt:
            params = {
                **self.model_training_parameters,
                **{
                    "n_estimators": hp.get("n_estimators"),
                    "learning_rate": hp.get("learning_rate"),
                    "reg_alpha": hp.get("reg_alpha"),
                    "reg_lambda": hp.get("reg_lambda"),
                },
            }
        else:
            params = self.model_training_parameters

        logger.info(f"Model training parameters : {params}")

        model = LGBMRegressor(**params)

        model.fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            init_model=lgbm_model,
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
                    f"Fit live predictions not warmed up yet. Still {abs(candle_diff)} candles to go"
                )
                warmed_up = False

        pred_df_full = (
            self.dd.historic_predictions[pair].tail(num_candles).reset_index(drop=True)
        )
        pred_df_sorted = pd.DataFrame()
        for label in pred_df_full.keys():
            if pred_df_full[label].dtype == object:
                continue
            pred_df_sorted[label] = pred_df_full[label]

        # pred_df_sorted = pred_df_sorted
        for col in pred_df_sorted:
            pred_df_sorted[col] = pred_df_sorted[col].sort_values(
                ascending=False, ignore_index=True
            )
        frequency = num_candles / (
            self.freqai_info["feature_parameters"]["label_period_candles"] * 2
        )
        max_pred = pred_df_sorted.iloc[: int(frequency)].mean()
        min_pred = pred_df_sorted.iloc[-int(frequency) :].mean()

        if not warmed_up:
            dk.data["extra_returns_per_train"]["&s-maxima_sort_threshold"] = 2
            dk.data["extra_returns_per_train"]["&s-minima_sort_threshold"] = -2
        else:
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
                self.freqai_info.get("weibull_outlier_threshold", 0.999), *f
            )

        dk.data["DI_value_mean"] = pred_df_full["DI_values"].mean()
        dk.data["DI_value_std"] = pred_df_full["DI_values"].std()
        dk.data["extra_returns_per_train"]["DI_value_param1"] = f[0]
        dk.data["extra_returns_per_train"]["DI_value_param2"] = f[1]
        dk.data["extra_returns_per_train"]["DI_value_param3"] = f[2]
        dk.data["extra_returns_per_train"]["DI_cutoff"] = cutoff

    def eval_set_and_weights(self, X_test, y_test, test_weights):
        if (
            self.freqai_info.get("data_split_parameters", {}).get(
                "test_size", TEST_SIZE
            )
            == 0
        ):
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(X_test, y_test)]
            eval_weights = [test_weights]

        return eval_set, eval_weights


def objective(trial, X, y, weights, X_test, y_test, params):
    study_params = {
        "objective": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
    }
    params = {**params, **study_params}
    window = trial.suggest_int("train_period_candles", 1152, 17280, step=300)

    # Fit the model
    model = LGBMRegressor(**params)
    X = X.tail(window)
    y = y.tail(window)
    weights = weights[-window:]
    model.fit(
        X,
        y,
        sample_weight=weights,
        eval_set=[(X_test, y_test)],
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, "rmse")],
    )
    y_pred = model.predict(X_test)

    error = sklearn.metrics.root_mean_squared_error(y_test, y_pred)

    return error
