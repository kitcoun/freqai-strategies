import copy
import gc
import json
import logging
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from statistics import stdev
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch as th
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.strategy import timeframe_to_minutes
from gymnasium.spaces import Box
from numpy.typing import NDArray
from optuna import Trial, TrialPruned, create_study
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import (
    BaseStorage,
    JournalStorage,
    RDBStorage,
    RetryFailedTrialCallback,
)
from optuna.storages.journal import JournalFileBackend
from optuna.study import Study, StudyDirection
from pandas import DataFrame, concat, merge
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
logger = logging.getLogger(__name__)


class ForceActions(IntEnum):
    Take_profit = 1
    Stop_loss = 2
    Timeout = 3


class ReforceXY(BaseReinforcementLearningModel):
    """
    Custom Freqtrade Freqai reinforcement learning prediction model.
    Model specific config:
    {
        "freqaimodel": "ReforceXY",
        "strategy": "RLAgentStrategy",
        "minimal_roi": {"0": 0.03},                 // Take_profit exit value used with force_actions
        "stoploss": -0.02,                          // Stop_loss exit value used with force_actions
        ...
        "freqai": {
            ...
            "rl_config": {
                ...
                "max_trade_duration_candles": 96,   // Timeout exit value used with force_actions
                "force_actions": false,             // Utilize minimal_roi, stoploss, and max_trade_duration_candles as TP/SL/Timeout in the environment
                "n_envs": 1,                        // Number of DummyVecEnv environments
                "frame_stacking": 0,                // Number of VecFrameStack stacks (set > 1 to use)
                "lr_schedule": false,               // Enable learning rate linear schedule
                "cr_schedule": false,               // Enable clip range linear schedule
                "max_no_improvement_evals": 0,      // Maximum consecutive evaluations without a new best model
                "min_evals": 0,                     // Number of evaluations before start to count evaluations without improvements
                "check_envs": true,                 // Check that an environment follows Gym API
                "plot_new_best": false,             // Enable tensorboard rollout plot upon finding a new best model
            },
            "rl_config_optuna": {
                "enabled": false,                   // Enable optuna hyperopt
                "per_pair: false,                   // Enable per pair hyperopt
                "n_trials": 100,
                "n_startup_trials": 15,
                "timeout_hours": 0,
            }
        }
    }
    Requirements:
        - pip install optuna

    Optional:
        - pip install optuna-dashboard
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI model requires StaticPairList method defined in pairlists configuration and pair_whitelist defined in exchange section configuration"
            )
        self.is_maskable: bool = (
            self.model_type == "MaskablePPO"
        )  # Enable action masking
        self.lr_schedule: bool = self.rl_config.get("lr_schedule", False)
        self.cr_schedule: bool = self.rl_config.get("cr_schedule", False)
        self.n_envs: int = self.rl_config.get("n_envs", 1)
        self.frame_stacking: int = self.rl_config.get("frame_stacking", 0)
        self.max_no_improvement_evals: int = self.rl_config.get(
            "max_no_improvement_evals", 0
        )
        self.min_evals: int = self.rl_config.get("min_evals", 0)
        self.plot_new_best: bool = self.rl_config.get("plot_new_best", False)
        self.check_envs: bool = self.rl_config.get("check_envs", True)
        self.progressbar_callback: Optional[ProgressBarCallback] = None
        # Optuna hyperopt
        self.rl_config_optuna: Dict[str, Any] = self.freqai_info.get(
            "rl_config_optuna", {}
        )
        self.hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.rl_config_optuna.get("enabled", False)
            and self.data_split_parameters.get("test_size", 0.1) > 0
        )
        self.optuna_timeout_hours: float = self.rl_config_optuna.get("timeout_hours", 0)
        self.optuna_n_trials: int = self.rl_config_optuna.get("n_trials", 100)
        self.optuna_n_startup_trials: int = self.rl_config_optuna.get(
            "n_startup_trials", 15
        )
        self.optuna_callback: Optional[MaskableTrialEvalCallback] = None
        self._model_params_cache: Optional[Dict[str, Any]] = None
        self.unset_unsupported()

    def unset_unsupported(self) -> None:
        """
        If user has activated any features that may conflict, this
        function will set them to proper values and warn them
        """
        if not isinstance(self.n_envs, int) or self.n_envs < 1:
            logger.warning("Invalid n_envs=%s. Forcing n_envs=1", self.n_envs)
            self.n_envs = 1
        if not isinstance(self.frame_stacking, int) or self.frame_stacking < 0:
            logger.warning(
                "Invalid frame_stacking=%s. Forcing frame_stacking=0",
                self.frame_stacking,
            )
            self.frame_stacking = 0
        if self.frame_stacking == 1:
            logger.warning(
                "Setting frame_stacking=%s is equivalent to no stacking; use >=2 or 0",
                self.frame_stacking,
            )
        if self.continual_learning and self.frame_stacking:
            logger.warning(
                "User tried to use continual_learning with frame_stacking. \
                Deactivating continual_learning"
            )
            self.continual_learning = False

    def set_train_and_eval_environments(
        self,
        data_dictionary: Dict[str, DataFrame],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
    ) -> None:
        """
        Set training and evaluation environments
        """
        self.close_envs()

        train_df = data_dictionary.get("train_features")
        test_df = data_dictionary.get("test_features")
        env_dict = self.pack_env_dict(dk.pair)
        seed = self.get_model_params().get("seed", 42)
        set_random_seed(seed)

        if self.check_envs:
            logger.info("Checking environments")
            _train_env_check = self.MyRLEnv(
                id="train_env_check", df=train_df, prices=prices_train, **env_dict
            )
            try:
                check_env(_train_env_check)
            finally:
                _train_env_check.close()
            _eval_env_check = self.MyRLEnv(
                id="eval_env_check", df=test_df, prices=prices_test, **env_dict
            )
            try:
                check_env(_eval_env_check)
            finally:
                _eval_env_check.close()

        logger.info("Populating environments: %s", self.n_envs)
        train_env = DummyVecEnv(
            [
                make_env(
                    self.MyRLEnv,
                    f"train_env{i}",
                    i,
                    seed,
                    train_df,
                    prices_train,
                    env_info=env_dict,
                )
                for i in range(self.n_envs)
            ]
        )
        eval_env = DummyVecEnv(
            [
                make_env(
                    self.MyRLEnv,
                    f"eval_env{i}",
                    i,
                    seed + 10_000,
                    test_df,
                    prices_test,
                    env_info=env_dict,
                )
                for i in range(self.n_envs)
            ]
        )

        if self.frame_stacking:
            logger.info("Frame stacking: %s", self.frame_stacking)
            train_env = VecFrameStack(train_env, n_stack=self.frame_stacking)
            eval_env = VecFrameStack(eval_env, n_stack=self.frame_stacking)

        self.train_env = VecMonitor(train_env)
        self.eval_env = VecMonitor(eval_env)

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        """
        if self._model_params_cache is not None:
            return copy.deepcopy(self._model_params_cache)

        model_params: Dict[str, Any] = copy.deepcopy(self.model_training_parameters)

        if model_params.get("seed") is None:
            model_params["seed"] = 42

        if not self.hyperopt and self.lr_schedule:
            lr = model_params.get("learning_rate", 0.0003)
            if isinstance(lr, (int, float)):
                lr = float(lr)
                model_params["learning_rate"] = linear_schedule(lr)
                logger.info(
                    "Learning rate linear schedule enabled, initial value: %s", lr
                )

        if not self.hyperopt and "PPO" in self.model_type and self.cr_schedule:
            cr = model_params.get("clip_range", 0.2)
            if isinstance(cr, (int, float)):
                cr = float(cr)
                model_params["clip_range"] = linear_schedule(cr)
                logger.info("Clip range linear schedule enabled, initial value: %s", cr)

        if "DQN" in self.model_type:
            if model_params.get("gradient_steps") is None:
                model_params["gradient_steps"] = compute_gradient_steps(
                    model_params.get("train_freq"), model_params.get("subsample_steps")
                )
            if "subsample_steps" in model_params:
                model_params.pop("subsample_steps", None)

        if not model_params.get("policy_kwargs"):
            model_params["policy_kwargs"] = {}

        default_net_arch: list[int] = [128, 128]
        net_arch: Union[
            list[int],
            Dict[str, list[int]],
            Literal["small", "medium", "large", "extra_large"],
        ] = model_params.get("policy_kwargs", {}).get("net_arch", default_net_arch)

        if "PPO" in self.model_type:
            if isinstance(net_arch, str):
                resolved_net_arch = get_net_arch(self.model_type, net_arch)
                if isinstance(resolved_net_arch, dict):
                    model_params["policy_kwargs"]["net_arch"] = resolved_net_arch
                else:
                    model_params["policy_kwargs"]["net_arch"] = {
                        "pi": resolved_net_arch,
                        "vf": resolved_net_arch,
                    }
            elif isinstance(net_arch, list):
                model_params["policy_kwargs"]["net_arch"] = {
                    "pi": net_arch,
                    "vf": net_arch,
                }
            elif isinstance(net_arch, dict):
                pi: Optional[list[int]] = net_arch.get("pi")
                vf: Optional[list[int]] = net_arch.get("vf")
                if not isinstance(pi, list) or not isinstance(vf, list):
                    model_params["policy_kwargs"]["net_arch"] = {
                        "pi": pi
                        if isinstance(pi, list)
                        else (vf if isinstance(vf, list) else default_net_arch),
                        "vf": vf
                        if isinstance(vf, list)
                        else (pi if isinstance(pi, list) else default_net_arch),
                    }
                else:
                    model_params["policy_kwargs"]["net_arch"] = net_arch
        else:
            if isinstance(net_arch, str):
                model_params["policy_kwargs"]["net_arch"] = get_net_arch(
                    self.model_type, net_arch
                )
            else:
                model_params["policy_kwargs"]["net_arch"] = net_arch

        model_params["policy_kwargs"]["activation_fn"] = get_activation_fn(
            model_params.get("policy_kwargs", {}).get("activation_fn", "relu")
        )
        model_params["policy_kwargs"]["optimizer_class"] = get_optimizer_class(
            model_params.get("policy_kwargs", {}).get("optimizer_class", "adam")
        )

        self._model_params_cache = model_params
        return copy.deepcopy(self._model_params_cache)

    def get_callbacks(
        self, eval_freq: int, data_path: str, trial: Optional[Trial] = None
    ) -> list[BaseCallback]:
        """
        Get the model specific callbacks
        """
        callbacks: list[BaseCallback] = []
        no_improvement_callback = None
        rollout_plot_callback = None
        verbose = int(self.get_model_params().get("verbose", 0))

        if self.plot_new_best:
            rollout_plot_callback = RolloutPlotCallback(verbose=verbose)

        if self.max_no_improvement_evals:
            no_improvement_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=self.max_no_improvement_evals,
                min_evals=self.min_evals,
                verbose=verbose,
            )

        if self.activate_tensorboard:
            info_callback = InfoMetricsCallback(actions=Actions, verbose=verbose)
            callbacks.append(info_callback)

        if self.rl_config.get("progress_bar", False):
            self.progressbar_callback = ProgressBarCallback()
            callbacks.append(self.progressbar_callback)

        self.eval_callback = MaskableEvalCallback(
            self.eval_env,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            best_model_save_path=data_path,
            use_masking=self.is_maskable,
            callback_on_new_best=rollout_plot_callback,
            callback_after_eval=no_improvement_callback,
            verbose=verbose,
        )
        callbacks.append(self.eval_callback)

        if not trial:
            return callbacks

        self.optuna_callback = MaskableTrialEvalCallback(
            self.eval_env,
            trial,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            best_model_save_path=data_path,
            use_masking=self.is_maskable,
            verbose=verbose,
        )
        callbacks.append(self.optuna_callback)
        return callbacks

    def fit(
        self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        if self.train_env is None or self.eval_env is None:
            raise RuntimeError("Environments not set. Cannot run model training")
        train_df = data_dictionary.get("train_features")
        train_timesteps = len(train_df)
        test_df = data_dictionary.get("test_features")
        test_timesteps = len(test_df)
        train_cycles = max(1, int(self.rl_config.get("train_cycles", 25)))
        total_timesteps = (
            (train_timesteps * train_cycles + self.n_envs - 1) // self.n_envs
        ) * self.n_envs
        train_days = steps_to_days(train_timesteps, self.config.get("timeframe"))
        test_days = steps_to_days(test_timesteps, self.config.get("timeframe"))
        total_days = steps_to_days(total_timesteps, self.config.get("timeframe"))

        logger.info("Model: %s", self.model_type)
        logger.info(
            "Train: %s steps (%s days), %s cycles, %s envs -> total %s steps (%s days)",
            train_timesteps,
            train_days,
            train_cycles,
            self.n_envs,
            total_timesteps,
            total_days,
        )
        logger.info("Test: %s steps (%s days)", test_timesteps, test_days)
        logger.info("Action masking: %s", self.is_maskable)
        logger.info("Hyperopt: %s", self.hyperopt)

        start_time = time.time()
        if self.hyperopt:
            best_trial_params = self.study(train_df, total_timesteps, dk)
            if best_trial_params is None:
                logger.error(
                    "Hyperopt failed. Using default configured model params instead"
                )
                best_trial_params = self.get_model_params()
            model_params = best_trial_params
        else:
            model_params = self.get_model_params()
        logger.info("%s params: %s", self.model_type, model_params)

        if self.activate_tensorboard:
            tensorboard_log_path = Path(
                self.full_path / "tensorboard" / dk.pair.split("/")[0]
            )
        else:
            tensorboard_log_path = None

        model = self.get_init_model(dk.pair)
        if model is not None:
            logger.info(
                "Continual training activated: starting training from previously trained model state"
            )
            model.set_env(self.train_env)
        else:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                tensorboard_log=tensorboard_log_path,
                **model_params,
            )

        callbacks = self.get_callbacks(
            max(1, train_timesteps // self.n_envs), str(dk.data_path)
        )
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            self.close_envs()
            if hasattr(model, "env") and model.env is not None:
                model.env.close()
        time_spent = time.time() - start_time
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        model_filename = dk.model_filename if dk.model_filename else "best"
        model_filepath = Path(dk.data_path / f"{model_filename}_model.zip")
        if model_filepath.is_file():
            logger.info("Found best model at %s", model_filepath)
            try:
                best_model = self.MODELCLASS.load(
                    dk.data_path / f"{model_filename}_model"
                )
                return best_model
            except Exception as e:
                logger.error(f"Error at loading best model: {repr(e)}", exc_info=True)

        logger.info(
            "Could not find best model at %s, using final model instead", model_filepath
        )

        return model

    def rl_model_predict(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any
    ) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :param dataframe: DataFrame = the dataframe of features to make the predictions on
        :param dk: FreqaiDatakitchen = data kitchen for the current pair
        :param model: Any = the trained model used to inference the features.
        """

        def _normalize_position(position: Any) -> Positions:
            if isinstance(position, Positions):
                return position
            try:
                f = float(position)
                if f == float(Positions.Long.value):
                    return Positions.Long
                if f == float(Positions.Short.value):
                    return Positions.Short
                return Positions.Neutral
            except Exception:
                return Positions.Neutral

        def _is_valid(action: int, position: Any) -> bool:
            """
            Determine if the action is valid for the step
            """
            position = _normalize_position(position)
            # Agent should only try to exit if it is in position
            if action in (Actions.Short_exit.value, Actions.Long_exit.value):
                if position not in (Positions.Short, Positions.Long):
                    return False

            # Agent should only try to enter if it is not in position
            if action in (Actions.Short_enter.value, Actions.Long_enter.value):
                if position != Positions.Neutral:
                    return False

            return True

        def _action_masks(position: Any) -> list[bool]:
            return [_is_valid(action.value, position) for action in Actions]

        def _predict(window) -> int:
            observation: DataFrame = dataframe.loc[window.index]
            action_masks_param: Dict[str, Any] = {}

            if self.rl_config.get("add_state_info", False):
                if self.live:
                    position, pnl, trade_duration = self.get_state_info(dk.pair)
                else:
                    # TODO: state info handling for backtests
                    position = Positions.Neutral
                    pnl = 0.0
                    trade_duration = 0

                # STATE_INFO
                observation["pnl"] = pnl
                observation["position"] = position
                observation["trade_duration"] = trade_duration

                if self.is_maskable:
                    action_masks_param = {"action_masks": _action_masks(position)}

            np_observation = observation.to_numpy(dtype=np.float32)

            frame_stacking = self.frame_stacking
            if frame_stacking and frame_stacking > 1:
                if not hasattr(_predict, "_frame_buffer"):
                    _predict._frame_buffer = []
                fb: list[np.ndarray] = getattr(_predict, "_frame_buffer")
                fb.append(np_observation)
                if len(fb) > frame_stacking:
                    del fb[0 : len(fb) - frame_stacking]
                if len(fb) < frame_stacking:
                    pad_needed = frame_stacking - len(fb)
                    fb_padded = [fb[0]] * pad_needed + fb
                else:
                    fb_padded = fb
                stacked_observations = np.concatenate(fb_padded, axis=1)
                observations = stacked_observations.reshape(1, -1)
            else:
                observations = np_observation.reshape(1, -1)

            action, _ = model.predict(
                observations, deterministic=True, **action_masks_param
            )
            return int(action)

        actions = dataframe.iloc[:, 0].rolling(window=self.CONV_WIDTH).apply(_predict)
        return DataFrame({label: actions for label in dk.label_list})

    def get_storage(self, pair: Optional[str] = None) -> BaseStorage:
        """
        Get the storage for Optuna
        """
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}" if pair else "optuna"
        storage_backend = self.rl_config_optuna.get("storage", "sqlite")
        if storage_backend == "sqlite":
            storage = RDBStorage(
                url=f"sqlite:///{storage_dir}/{storage_filename}.sqlite",
                heartbeat_interval=60,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            )
        elif storage_backend == "file":
            storage = JournalStorage(
                JournalFileBackend(f"{storage_dir}/{storage_filename}.log")
            )
        else:
            raise ValueError(
                f"Unsupported storage backend: {storage_backend}. Supported backends are: 'sqlite' and 'file'"
            )
        return storage

    @staticmethod
    def study_has_best_trial(study: Optional[Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trial
            return True
        except (ValueError, KeyError):
            return False

    def study(
        self, train_df: DataFrame, total_timesteps: int, dk: FreqaiDataKitchen
    ) -> Optional[Dict[str, Any]]:
        """
        Runs hyperparameter optimization using Optuna and returns the best hyperparameters found merged with the user defined parameters
        """
        identifier = self.freqai_info.get("identifier", "no_id_provided")
        study_name = (
            f"{identifier}-{dk.pair}"
            if self.rl_config_optuna.get("per_pair", False)
            else identifier
        )
        storage = (
            self.get_storage(dk.pair)
            if self.rl_config_optuna.get("per_pair", False)
            else self.get_storage()
        )
        eval_freq = max(1, len(train_df) // self.n_envs)
        max_resource = max(1, (total_timesteps + eval_freq - 1) // eval_freq)
        min_resource = min(3, max_resource)
        study: Study = create_study(
            study_name=study_name,
            sampler=TPESampler(
                n_startup_trials=self.optuna_n_startup_trials,
                multivariate=True,
                group=True,
                seed=self.rl_config_optuna.get("seed", 42),
            ),
            pruner=HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=3,
            ),
            direction=StudyDirection.MAXIMIZE,
            storage=storage,
            load_if_exists=True,
        )
        hyperopt_failed = False
        start_time = time.time()
        try:
            study.optimize(
                lambda trial: self.objective(trial, train_df, total_timesteps, dk),
                n_trials=self.optuna_n_trials,
                timeout=(
                    hours_to_seconds(self.optuna_timeout_hours)
                    if self.optuna_timeout_hours
                    else None
                ),
                gc_after_trial=True,
                show_progress_bar=self.rl_config.get("progress_bar", False),
                n_jobs=1,
            )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"Hyperopt {study_name} failed ({time_spent:.2f} secs): {repr(e)}",
                exc_info=True,
            )
            hyperopt_failed = True
        time_spent = time.time() - start_time
        if not ReforceXY.study_has_best_trial(study):
            logger.error(
                f"Hyperopt {study_name} failed ({time_spent:.2f} secs): no study best trial found"
            )
            hyperopt_failed = True

        if hyperopt_failed:
            best_trial_params = self.load_best_trial_params(
                dk.pair if self.rl_config_optuna.get("per_pair", False) else None
            )
            if best_trial_params is None:
                logger.error(
                    f"Hyperopt {study_name} failed ({time_spent:.2f} secs): no previously saved best trial params found"
                )
                return None
        else:
            best_trial_params = study.best_trial.params

        logger.info(
            "------------ Hyperopt %s results (%.2f secs) ------------",
            study_name,
            time_spent,
        )
        if ReforceXY.study_has_best_trial(study):
            logger.info(
                "Best trial: %s. Score: %s",
                study.best_trial.number,
                study.best_trial.value,
            )
        logger.info("Best trial params: %s", best_trial_params)
        logger.info("-------------------------------------------------------")

        self.save_best_trial_params(
            best_trial_params,
            dk.pair if self.rl_config_optuna.get("per_pair", False) else None,
        )

        return deepmerge(
            self.get_model_params(),
            convert_optuna_params_to_model_params(self.model_type, best_trial_params),
        )

    def save_best_trial_params(
        self, best_trial_params: Dict[str, Any], pair: Optional[str] = None
    ) -> None:
        """
        Save the best trial hyperparameters found during hyperparameter optimization
        """
        best_trial_params_filename = (
            f"hyperopt-best-params-{pair.split('/')[0]}"
            if pair
            else "hyperopt-best-params"
        )
        best_trial_params_path = Path(
            self.full_path / f"{best_trial_params_filename}.json"
        )
        log_msg: str = (
            f"{pair}: saving best params to {best_trial_params_path} JSON file"
            if pair
            else f"Saving best params to {best_trial_params_path} JSON file"
        )
        logger.info(log_msg)
        try:
            with best_trial_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(best_trial_params, write_file, indent=4)
        except Exception as e:
            logger.error(
                f"Error saving best trial params to {best_trial_params_path}: {repr(e)}",
                exc_info=True,
            )
            raise

    def load_best_trial_params(
        self, pair: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best trial hyperparameters found and saved during hyperparameter optimization
        """
        best_trial_params_filename = (
            f"hyperopt-best-params-{pair.split('/')[0]}"
            if pair
            else "hyperopt-best-params"
        )
        best_trial_params_path = Path(
            self.full_path / f"{best_trial_params_filename}.json"
        )
        log_msg: str = (
            f"{pair}: loading best params from {best_trial_params_path} JSON file"
            if pair
            else f"Loading best params from {best_trial_params_path} JSON file"
        )
        if best_trial_params_path.is_file():
            logger.info(log_msg)
            with best_trial_params_path.open("r", encoding="utf-8") as read_file:
                best_trial_params = json.load(read_file)
            return best_trial_params
        return None

    def objective(
        self,
        trial: Trial,
        train_df: DataFrame,
        total_timesteps: int,
        dk: FreqaiDataKitchen,
    ) -> float:
        """
        Defines a single trial for hyperparameter optimization using Optuna
        """
        if self.train_env is None or self.eval_env is None:
            raise RuntimeError("Environments not set. Cannot run HPO trial")
        if "PPO" in self.model_type:
            params = sample_params_ppo(trial, self.n_envs)
            if params.get("n_steps", 0) > total_timesteps:
                raise TrialPruned("n_steps is greater than total_timesteps")
        elif "QRDQN" in self.model_type:
            params = sample_params_qrdqn(trial)
        elif "DQN" in self.model_type:
            params = sample_params_dqn(trial)
        else:
            raise NotImplementedError

        # Ensure that the sampled parameters take precedence
        params = deepmerge(self.get_model_params(), params)

        nan_encountered = False

        if self.activate_tensorboard:
            tensorboard_log_path = Path(
                self.full_path / "tensorboard" / dk.pair.split("/")[0]
            )
        else:
            tensorboard_log_path = None

        logger.info("------------ Hyperopt trial %d ------------", trial.number)
        logger.info("Trial %s params: %s", trial.number, params)

        model = self.MODELCLASS(
            self.policy_type,
            self.train_env,
            tensorboard_log=tensorboard_log_path,
            **params,
        )

        callbacks = self.get_callbacks(
            max(1, len(train_df) // self.n_envs), str(dk.data_path), trial
        )
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except AssertionError:
            logger.warning("Optuna encountered NaN (AssertionError)")
            nan_encountered = True
        except ValueError as e:
            if "NaN" in str(e):
                logger.warning("Optuna encountered NaN (ValueError)")
                nan_encountered = True
            else:
                raise
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()

        if nan_encountered:
            raise TrialPruned("NaN encountered during training")

        if self.optuna_callback.is_pruned:
            raise TrialPruned()

        return self.optuna_callback.best_mean_reward

    def close_envs(self) -> None:
        """
        Closes the training and evaluation environments if they are open
        """
        if self.train_env:
            try:
                self.train_env.close()
            finally:
                self.train_env = None
        if self.eval_env:
            try:
                self.eval_env.close()
            finally:
                self.eval_env = None

    class MyRLEnv(Base5ActionRLEnv):
        """
        Env
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._set_observation_space()
            self.force_actions: bool = self.rl_config.get("force_actions", False)
            self._force_action: Optional[ForceActions] = None
            self.take_profit: float = self.config.get("minimal_roi", {}).get("0", 0.03)
            self.stop_loss: float = self.config.get("stoploss", -0.02)
            self.timeout: int = self.rl_config.get("max_trade_duration_candles", 128)
            self._last_closed_position: Optional[Positions] = None
            self._last_closed_trade_tick: int = 0
            if self.force_actions:
                logger.info(
                    "%s - take_profit: %s, stop_loss: %s, timeout: %s candles (%s days), observation_space: %s",
                    self.id,
                    self.take_profit,
                    self.stop_loss,
                    self.timeout,
                    steps_to_days(self.timeout, self.config.get("timeframe")),
                    self.observation_space,
                )

        def _set_observation_space(self) -> None:
            """
            Set the observation space
            """
            signal_features = self.signal_features.shape[1]
            if self.add_state_info:
                # STATE_INFO
                self.state_features = ["pnl", "position", "trade_duration"]
                self.total_features = signal_features + len(self.state_features)
            else:
                self.state_features = []
                self.total_features = signal_features

            self.shape = (self.window_size, self.total_features)
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
            )

        def reset_env(
            self,
            df: DataFrame,
            prices: DataFrame,
            window_size: int,
            reward_kwargs: Dict[str, Any],
            starting_point=True,
        ) -> None:
            """
            Resets the environment when the agent fails
            """
            super().reset_env(df, prices, window_size, reward_kwargs, starting_point)
            self._set_observation_space()

        def reset(
            self, seed=None, **kwargs
        ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
            """
            Reset is called at the beginning of every episode
            """
            observation, history = super().reset(seed, **kwargs)
            self._force_action: Optional[ForceActions] = None
            self._last_closed_position: Optional[Positions] = None
            self._last_closed_trade_tick: int = 0
            return observation, history

        def _get_reward_factor_at_trade_exit(
            self,
            factor: float,
            pnl: float,
            trade_duration: int,
            max_trade_duration: int,
        ) -> float:
            """
            Compute the reward factor at trade exit
            """
            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5
            if pnl > self.profit_aim * self.rr:
                factor *= float(
                    self.rl_config.get("model_reward_parameters", {}).get(
                        "win_reward_factor", 2.0
                    )
                )
            return factor

        def calculate_reward(self, action: int) -> float:
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.

            Warning!
            This is function is a showcase of functionality designed to show as many possible
            environment control features as possible. It is also designed to run quickly
            on small computers. This is a benchmark, it is *not* for live production.

            :param action: int = The action made by the agent for the current candle.
            :return:
            float = the reward to give to the agent for current step (used for optimization
                    of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._force_action and not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return self.rl_config.get("model_reward_parameters", {}).get(
                    "invalid_action", -2.0
                )

            pnl = self.get_unrealized_profit()
            # mrr = self.get_most_recent_return()
            # mrp = self.get_most_recent_profit()

            max_trade_duration = max(1, self.timeout)
            trade_duration = self.get_trade_duration()

            factor = 100.0

            # Force exits
            if self._force_action in (
                ForceActions.Take_profit,
                ForceActions.Stop_loss,
                ForceActions.Timeout,
            ):
                return pnl * self._get_reward_factor_at_trade_exit(
                    factor, pnl, trade_duration, max_trade_duration
                )

            # # you can use feature values from dataframe
            # rsi_now = self.get_feature_value(
            #     name="%-rsi",
            #     period=8,
            #     pair=self.pair,
            #     timeframe=self.config.get("timeframe"),
            #     raw=True,
            # )

            # # reward agent for entering trades when RSI is low
            # if (
            #     action in (Actions.Long_enter.value, Actions.Short_enter.value)
            #     and self._position == Positions.Neutral
            # ):
            #     if rsi_now < 40:
            #         factor = 40 / rsi_now
            #     else:
            #         factor = 1
            #     return 25.0 * factor

            # discourage agent from sitting idle too long
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                idle_duration = self.get_idle_duration()
                return float(-0.01 * idle_duration**1.05)

            # pnl and duration aware agent reward while sitting in position
            if (
                self._position in (Positions.Short, Positions.Long)
                and action == Actions.Neutral.value
            ):
                duration_fraction = trade_duration / max_trade_duration
                max_pnl = max(self.get_most_recent_max_pnl(), pnl)

                if max_pnl > 0:
                    drawdown_penalty = (
                        0.0025 * factor * (max_pnl - pnl) * duration_fraction
                    )
                else:
                    drawdown_penalty = 0.0

                lambda1 = 0.05
                lambda2 = 0.1
                if pnl >= 0:
                    if duration_fraction <= 1.0:
                        duration_penalty_factor = 1.0
                    else:
                        duration_penalty_factor = 1.0 / (
                            1.0 + lambda1 * (duration_fraction - 1.0)
                        )
                    return (
                        factor * pnl * duration_penalty_factor
                        - lambda2 * duration_fraction
                        - drawdown_penalty
                    )
                else:
                    return (
                        factor * pnl * (1.0 + lambda1 * duration_fraction)
                        - 2.0 * lambda2 * duration_fraction
                        - drawdown_penalty
                    )

            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                return pnl * self._get_reward_factor_at_trade_exit(
                    factor, pnl, trade_duration, max_trade_duration
                )

            # close short
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                return pnl * self._get_reward_factor_at_trade_exit(
                    factor, pnl, trade_duration, max_trade_duration
                )

            return 0.0

        def _get_observation(self) -> NDArray[np.float32]:
            """
            This may or may not be independent of action types, user can inherit
            this in their custom "MyRLEnv"
            """
            features_window = self.signal_features[
                (self._current_tick - self.window_size) : self._current_tick
            ]
            if self.add_state_info:
                features_and_state = DataFrame(
                    np.zeros(
                        (len(features_window), len(self.state_features)),
                        dtype=np.float32,
                    ),
                    columns=self.state_features,
                    index=features_window.index,
                )
                # STATE_INFO
                features_and_state["pnl"] = self.get_unrealized_profit()
                features_and_state["position"] = self._position.value
                features_and_state["trade_duration"] = self.get_trade_duration()

                features_and_state = concat(
                    [features_window, features_and_state], axis=1
                )
                return features_and_state.to_numpy(dtype=np.float32)
            else:
                return features_window.to_numpy(dtype=np.float32)

        def _get_force_action(self) -> Optional[ForceActions]:
            if not self.force_actions or self._position == Positions.Neutral:
                return None

            trade_duration = self.get_trade_duration()
            if trade_duration <= 1:
                return None
            if trade_duration >= self.timeout:
                return ForceActions.Timeout

            pnl = self.get_unrealized_profit()
            if pnl >= self.take_profit:
                return ForceActions.Take_profit
            if pnl <= self.stop_loss:
                return ForceActions.Stop_loss
            return None

        def _get_new_position(self, action: int) -> Positions:
            return {
                Actions.Long_enter.value: Positions.Long,
                Actions.Short_enter.value: Positions.Short,
            }[action]

        def _enter_trade(self, action: int) -> None:
            self._position = self._get_new_position(action)
            self._last_trade_tick = self._current_tick

        def _exit_trade(self) -> None:
            self._update_total_profit()
            self._last_closed_position = self._position
            self._position = Positions.Neutral
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = None

        def execute_trade(self, action: int) -> None:
            """
            Execute trade based on the given action
            """
            # Force exit trade
            if self._force_action:
                self._exit_trade()  # Exit trade due to force action
                self.append_trade_history(f"{self._force_action.name}")
                self.tensorboard_log(
                    f"{self._force_action.name}", category="actions/force"
                )
                return None

            if not self.is_tradesignal(action):
                return None

            # Enter trade based on action
            if action in (Actions.Long_enter.value, Actions.Short_enter.value):
                self._enter_trade(action)
                self.append_trade_history(f"{self._position.name}_enter")

            # Exit trade based on action
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                self._exit_trade()
                self.append_trade_history(f"{self._last_closed_position.name}_exit")

        def step(
            self, action: int
        ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
            """
            Take a step in the environment based on the provided action
            """
            self.tensorboard_log(Actions._member_names_[action], category="actions")
            self._current_tick += 1
            self._update_unrealized_total_profit()
            self._force_action = self._get_force_action()
            reward = self.calculate_reward(action)
            self.total_reward += reward
            info = {
                "tick": self._current_tick,
                "position": self._position.value,
                "action": action,
                "force_action": (
                    self._force_action.name if self._force_action else None
                ),
                "pnl": self.get_unrealized_profit(),
                "reward": round(reward, 5),
                "total_reward": round(self.total_reward, 5),
                "total_profit": round(self._total_profit, 5),
                "idle_duration": self.get_idle_duration(),
                "trade_duration": self.get_trade_duration(),
                "trade_count": len(self.trade_history),
            }
            self.execute_trade(action)
            info["position"] = self._position.value
            info["pnl"] = self.get_unrealized_profit()
            info["trade_duration"] = self.get_trade_duration()
            info["trade_count"] = len(self.trade_history)
            self._position_history.append(self._position)
            self._update_history(info)
            return (
                self._get_observation(),
                reward,
                self.is_terminated(),
                self.is_truncated(),
                info,
            )

        def append_trade_history(self, trade_type: str) -> None:
            self.trade_history.append(
                {
                    "tick": self._current_tick,
                    "type": trade_type.lower(),
                    "price": self.current_price(),
                    "profit": self.get_unrealized_profit(),
                }
            )

        def is_terminated(self) -> bool:
            return bool(
                self._current_tick == self._end_tick
                or self._total_profit <= self.max_drawdown
                or self._total_unrealized_profit <= self.max_drawdown
            )

        def is_truncated(self) -> bool:
            return False

        def is_tradesignal(self, action: int) -> bool:
            """
            Determine if the action is entry or exit
            """
            return (
                (
                    action in (Actions.Short_enter.value, Actions.Long_enter.value)
                    and self._position == Positions.Neutral
                )
                or (
                    action == Actions.Long_exit.value
                    and self._position == Positions.Long
                )
                or (
                    action == Actions.Short_exit.value
                    and self._position == Positions.Short
                )
            )

        def get_feature_value(
            self,
            name: str,
            period: int = 0,
            shift: int = 0,
            pair: str = "",
            timeframe: str = "",
            raw: bool = True,
        ) -> float:
            """
            Get feature value
            """
            feature_parts = [name]
            if period:
                feature_parts.append(f"_{period}")
            if shift:
                feature_parts.append(f"_shift-{shift}")
            if pair:
                feature_parts.append(f"_{pair.replace(':', '')}")
            if timeframe:
                feature_parts.append(f"_{timeframe}")
            feature_col = "".join(feature_parts)

            if not raw:
                return self.signal_features[feature_col].iloc[self._current_tick]
            return self.raw_features[feature_col].iloc[self._current_tick]

        def get_idle_duration(self) -> int:
            if self._position != Positions.Neutral:
                return 0
            if not self._last_closed_trade_tick:
                return self._current_tick - self._start_tick
            return self._current_tick - self._last_closed_trade_tick

        def get_most_recent_max_pnl(self) -> float:
            pnl_history = self.history.get("pnl")
            return (
                np.max(pnl_history) if pnl_history and len(pnl_history) > 0 else -np.inf
            )

        def get_most_recent_return(self) -> float:
            """
            Calculate the tick to tick return if in a trade.
            Return is generated from rising prices in Long
            and falling prices in Short positions.
            The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
            """
            if self._current_tick <= 0:
                return 0.0
            if self._position == Positions.Long:
                current_price = self.current_price()
                previous_price = self.previous_price()
                if (
                    self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1]
                    == Positions.Neutral
                ):
                    previous_price = self.add_entry_fee(previous_price)
                return np.log(current_price) - np.log(previous_price)
            if self._position == Positions.Short:
                current_price = self.current_price()
                previous_price = self.previous_price()
                if (
                    self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1]
                    == Positions.Neutral
                ):
                    previous_price = self.add_exit_fee(previous_price)
                return np.log(previous_price) - np.log(current_price)
            return 0.0

        def update_portfolio_log_returns(self):
            self.portfolio_log_returns[self._current_tick] = (
                self.get_most_recent_return()
            )

        def get_most_recent_profit(self) -> float:
            """
            Calculate the tick to tick unrealized profit if in a trade
            """
            if self._position == Positions.Long:
                current_price = self.add_exit_fee(self.current_price())
                previous_price = self.add_entry_fee(self.previous_price())
                return (current_price - previous_price) / previous_price
            elif self._position == Positions.Short:
                current_price = self.add_entry_fee(self.current_price())
                previous_price = self.add_exit_fee(self.previous_price())
                return (previous_price - current_price) / previous_price
            return 0.0

        def previous_price(self) -> float:
            return self.prices.iloc[self._current_tick - 1].get("open")

        def get_env_history(self) -> DataFrame:
            """
            Get environment data from the first to the last trade
            """
            if not self.history or not self.trade_history:
                logger.warning("history or trade_history is empty")
                return DataFrame()

            _history_df = DataFrame.from_dict(self.history)
            _trade_history_df = DataFrame.from_dict(self.trade_history)

            if (
                "tick" not in _history_df.columns
                or "tick" not in _trade_history_df.columns
            ):
                logger.warning("'tick' column is missing from history or trade_history")
                return DataFrame()

            _rollout_history = merge(
                _history_df, _trade_history_df, on="tick", how="left"
            ).ffill()
            _price_history = (
                self.prices.iloc[_rollout_history.tick].copy().reset_index()
            )
            history = merge(
                _rollout_history, _price_history, left_index=True, right_index=True
            )
            return history

        def get_env_plot(self) -> plt.Figure:
            """
            Plot trades and environment data
            """

            def transform_y_offset(ax, offset):
                return mtransforms.offset_copy(ax.transData, fig=fig, y=offset)

            def plot_markers(ax, ticks, marker, color, size, offset):
                ax.plot(
                    ticks,
                    marker=marker,
                    color=color,
                    markersize=size,
                    fillstyle="full",
                    transform=transform_y_offset(ax, offset),
                    linestyle="none",
                )

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                nrows=5,
                ncols=1,
                figsize=(14, 8),
                height_ratios=[5, 1, 1, 1, 1],
                sharex=True,
            )

            if len(self.trade_history) == 0:
                return fig

            history = self.get_env_history()
            if len(history) == 0:
                return fig

            history_price = history.get("price")
            if history_price is None or len(history_price) == 0:
                return fig
            history_type = history.get("type")
            if history_type is None or len(history_type) == 0:
                return fig
            history_open = history.get("open")
            if history_open is None or len(history_open) == 0:
                return fig

            enter_long_prices = history.loc[history_type == "long_enter", "price"]
            enter_short_prices = history.loc[history_type == "short_enter", "price"]
            exit_long_prices = history.loc[history_type == "long_exit", "price"]
            exit_short_prices = history.loc[history_type == "short_exit", "price"]
            take_profit_prices = history.loc[history_type == "take_profit", "price"]
            stop_loss_prices = history.loc[history_type == "stop_loss", "price"]
            timeout_prices = history.loc[history_type == "timeout", "price"]

            axs[0].plot(history_open, linewidth=1, color="orchid")

            plot_markers(axs[0], enter_long_prices, "^", "forestgreen", 5, -0.1)
            plot_markers(axs[0], enter_short_prices, "v", "firebrick", 5, 0.1)
            plot_markers(axs[0], exit_long_prices, ".", "cornflowerblue", 4, 0.1)
            plot_markers(axs[0], exit_short_prices, ".", "thistle", 4, -0.1)
            plot_markers(axs[0], take_profit_prices, "*", "lime", 8, 0.1)
            plot_markers(axs[0], stop_loss_prices, "x", "red", 8, -0.1)
            plot_markers(axs[0], timeout_prices, "1", "yellow", 8, 0.0)

            axs[1].set_ylabel("pnl")
            axs[1].plot(history.get("pnl"), linewidth=1, color="gray")
            axs[1].axhline(y=0, label="0", alpha=0.33, color="gray")
            axs[1].axhline(y=self.take_profit, label="tp", alpha=0.33, color="green")
            axs[1].axhline(y=self.stop_loss, label="sl", alpha=0.33, color="red")

            axs[2].set_ylabel("reward")
            axs[2].plot(history.get("reward"), linewidth=1, color="gray")
            axs[2].axhline(y=0, label="0", alpha=0.33)

            axs[3].set_ylabel("total_profit")
            axs[3].plot(history.get("total_profit"), linewidth=1, color="gray")
            axs[3].axhline(y=1, label="0", alpha=0.33)

            axs[4].set_ylabel("total_reward")
            axs[4].plot(history.get("total_reward"), linewidth=1, color="gray")
            axs[4].axhline(y=0, label="0", alpha=0.33)
            axs[4].set_xlabel("tick")

            _borders = ["top", "right", "bottom", "left"]
            for _ax in axs:
                for _border in _borders:
                    _ax.spines[_border].set_color("#5b5e4b")

            fig.suptitle(
                f"Total Reward: {self.total_reward:.2f} ~ "
                + f"Total Profit: {self._total_profit:.2f} ~ "
                + f"Trades: {len(self.trade_history)}"
            )
            fig.tight_layout()
            return fig

        def close(self) -> None:
            plt.close()
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()


class InfoMetricsCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def _on_training_start(self) -> None:
        _lr = self.model.learning_rate
        _lr = (
            float(_lr) if isinstance(_lr, (int, float, np.floating)) else "lr_schedule"
        )
        n_stack = 1
        training_env = getattr(self, "training_env", None)
        while training_env is not None:
            if hasattr(training_env, "n_stack"):
                try:
                    n_stack = int(getattr(training_env, "n_stack"))
                except Exception:
                    pass
                break
            training_env = getattr(training_env, "venv", None)
        hparam_dict: Dict[str, Any] = {
            "algorithm": self.model.__class__.__name__,
            "n_envs": int(self.model.n_envs),
            "n_stack": n_stack,
            "learning_rate": _lr,
            "gamma": float(self.model.gamma),
            "batch_size": int(self.model.batch_size),
        }
        if "PPO" in self.model.__class__.__name__:
            _cr = self.model.clip_range
            _cr = (
                float(_cr)
                if isinstance(_cr, (int, float, np.floating))
                else "cr_schedule"
            )
            hparam_dict.update(
                {
                    "clip_range": _cr,
                    "gae_lambda": float(self.model.gae_lambda),
                    "n_steps": int(self.model.n_steps),
                    "n_epochs": int(self.model.n_epochs),
                    "ent_coef": float(self.model.ent_coef),
                    "vf_coef": float(self.model.vf_coef),
                    "max_grad_norm": float(self.model.max_grad_norm),
                }
            )
            if getattr(self.model, "target_kl", None) is not None:
                hparam_dict["target_kl"] = float(self.model.target_kl)
        if "DQN" in self.model.__class__.__name__:
            hparam_dict.update(
                {
                    "buffer_size": int(self.model.buffer_size),
                    "gradient_steps": int(self.model.gradient_steps),
                    "learning_starts": int(self.model.learning_starts),
                    "target_update_interval": int(self.model.target_update_interval),
                    "exploration_initial_eps": float(
                        self.model.exploration_initial_eps
                    ),
                    "exploration_final_eps": float(self.model.exploration_final_eps),
                    "exploration_fraction": float(self.model.exploration_fraction),
                    "exploration_rate": float(self.model.exploration_rate),
                }
            )
            if "QRDQN" in self.model.__class__.__name__:
                hparam_dict.update({"n_quantiles": int(self.model.n_quantiles)})
        metric_dict = {
            "info/total_reward": 0.0,
            "info/total_profit": 1.0,
            "info/trade_count": 0,
            "info/trade_duration": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        infos_list: list[Dict[str, Any]] | None = self.locals.get("infos")
        aggregated_info: Dict[str, Any] = {}

        if isinstance(infos_list, list) and infos_list:
            numeric_acc: Dict[str, list[float]] = defaultdict(list)
            non_numeric_acc: Dict[str, set[Any]] = defaultdict(set)

            for info_dict in infos_list:
                if not isinstance(info_dict, dict):
                    continue
                for k, v in info_dict.items():
                    if k in {"episode", "terminal_observation", "TimeLimit.truncated"}:
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        numeric_acc[k].append(float(v))
                    else:
                        non_numeric_acc[k].add(v)

            for k, values in numeric_acc.items():
                if not values:
                    continue
                values_mean = sum(values) / len(values)
                aggregated_info[k] = values_mean
                if len(values) > 1:
                    try:
                        aggregated_info[f"{k}_std"] = stdev(values)
                    except Exception:
                        pass

            for key in ("reward", "pnl"):
                values = numeric_acc.get(key)
                if values:
                    aggregated_info[f"{key}_min"] = float(min(values))
                    aggregated_info[f"{key}_max"] = float(max(values))

            for k, values in non_numeric_acc.items():
                aggregated_info[k] = next(iter(values)) if len(values) == 1 else "mixed"

        if self.training_env is None:
            return True

        try:
            tensorboard_metrics_list = self.training_env.get_attr("tensorboard_metrics")
            tensorboard_metrics = (
                tensorboard_metrics_list[0] if tensorboard_metrics_list else {}
            )
        except Exception:
            tensorboard_metrics = {}

        for metric, value in aggregated_info.items():
            self.logger.record(f"info/{metric}", value)

        for category, metrics in tensorboard_metrics.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    self.logger.record(f"{category}/{metric}", value)
        return True


class RolloutPlotCallback(BaseCallback):
    """
    Tensorboard plot callback
    """

    def record_env(self) -> bool:
        figures = self.training_env.env_method("get_env_plot")
        for i, fig in enumerate(figures):
            figure = Figure(fig, close=True)
            self.logger.record(
                f"best/train_env_{i}", figure, exclude=("stdout", "log", "json", "csv")
            )
        return True

    def _on_step(self) -> bool:
        return self.record_env()


class MaskableTrialEvalCallback(MaskableEvalCallback):
    """
    Optuna maskable trial eval callback
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes: int = 10,
        eval_freq: int = 2048,
        deterministic: bool = True,
        use_masking: bool = True,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            use_masking=use_masking,
            **kwargs,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.is_pruned:
            return False
        _super_on_step = super()._on_step()
        if not _super_on_step:
            return False
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            last_mean_reward = getattr(self, "last_mean_reward", np.nan)
            if not isinstance(last_mean_reward, (int, float)) or not np.isfinite(
                float(last_mean_reward)
            ):
                self.is_pruned = True
                return False
            if hasattr(self.trial, "report"):
                try:
                    self.trial.report(last_mean_reward, self.eval_idx)
                except Exception:
                    self.is_pruned = True
                    return False
            if hasattr(self.trial, "should_prune") and self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def make_env(
    MyRLEnv: Type[BaseEnvironment],
    env_id: str,
    rank: int,
    seed: int,
    train_df: DataFrame,
    price: DataFrame,
    env_info: Dict[str, Any],
) -> Callable[[], BaseEnvironment]:
    """
    Utility function for multiprocessed env.

    :param MyRLEnv: (Type[BaseEnvironment]) environment class to instantiate
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param train_df: (DataFrame) feature dataframe for the environment
    :param price: (DataFrame) aligned price dataframe
    :param env_info: (dict) all required arguments to instantiate the environment
    :return:
    (Callable[[], BaseEnvironment]) closure that when called instantiates and returns the environment
    """

    def _init() -> BaseEnvironment:
        return MyRLEnv(
            df=train_df, prices=price, id=env_id, seed=seed + rank, **env_info
        )

    return _init


def deepmerge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts without mutating inputs"""
    dst_copy = copy.deepcopy(dst)
    for k, v in src.items():
        if (
            k in dst_copy
            and isinstance(dst_copy[k], Mapping)
            and isinstance(v, Mapping)
        ):
            dst_copy[k] = deepmerge(dst_copy[k], v)
        else:
            dst_copy[k] = v
    return dst_copy


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def compute_gradient_steps(train_freq: Any, subsample_steps: Any) -> int:
    tf: Optional[int] = None
    if isinstance(train_freq, (tuple, list)) and train_freq:
        tf = train_freq[0] if isinstance(train_freq[0], int) else None
    elif isinstance(train_freq, int):
        tf = train_freq

    ss: Optional[int] = subsample_steps if isinstance(subsample_steps, int) else None

    if isinstance(tf, int) and tf > 0 and isinstance(ss, int) and ss > 0:
        return min(tf, max(tf // ss, 1))
    return -1


@lru_cache(maxsize=32)
def hours_to_seconds(hours: float) -> float:
    """
    Converts hours to seconds
    """
    seconds = hours * 3600
    return seconds


@lru_cache(maxsize=32)
def steps_to_days(steps: int, timeframe: str) -> float:
    """
    Calculate the number of days based on the given number of steps
    """
    total_minutes = steps * timeframe_to_minutes(timeframe)
    days = total_minutes / (24 * 60)
    return round(days, 1)


def get_net_arch(
    model_type: str, net_arch_type: Literal["small", "medium", "large", "extra_large"]
) -> Union[list[int], Dict[str, list[int]]]:
    """
    Get network architecture
    """
    if "PPO" in model_type:
        return {
            "small": {"pi": [128, 128], "vf": [128, 128]},
            "medium": {"pi": [256, 256], "vf": [256, 256]},
            "large": {"pi": [512, 512], "vf": [512, 512]},
            "extra_large": {"pi": [1024, 1024], "vf": [1024, 1024]},
        }.get(net_arch_type, {"pi": [128, 128], "vf": [128, 128]})
    return {
        "small": [128, 128],
        "medium": [256, 256],
        "large": [512, 512],
        "extra_large": [1024, 1024],
    }.get(net_arch_type, [128, 128])


def get_activation_fn(
    activation_fn_name: Literal["tanh", "relu", "elu", "leaky_relu"],
) -> type[th.nn.Module]:
    """
    Get activation function
    """
    return {
        "tanh": th.nn.Tanh,
        "relu": th.nn.ReLU,
        "elu": th.nn.ELU,
        "leaky_relu": th.nn.LeakyReLU,
    }.get(activation_fn_name, th.nn.ReLU)


def get_optimizer_class(
    optimizer_class_name: Literal["adam", "adamw"],
) -> type[th.optim.Optimizer]:
    """
    Get optimizer class
    """
    return {
        "adam": th.optim.Adam,
        "adamw": th.optim.AdamW,
    }.get(optimizer_class_name, th.optim.Adam)


def convert_optuna_params_to_model_params(
    model_type: str, optuna_params: Dict[str, Any]
) -> Dict[str, Any]:
    model_params: Dict[str, Any] = {}
    policy_kwargs: Dict[str, Any] = {}

    lr = optuna_params.get("learning_rate")
    if lr is None:
        raise ValueError(f"missing 'learning_rate' in optuna params for {model_type}")
    lr: float | Callable[[float], float] = float(lr)
    if optuna_params.get("lr_schedule") == "linear":
        lr = linear_schedule(lr)

    if "PPO" in model_type:
        cr = optuna_params.get("clip_range")
        if cr is None:
            raise ValueError(f"missing 'clip_range' in optuna params for {model_type}")
        cr: float | Callable[[float], float] = float(cr)
        if optuna_params.get("cr_schedule") == "linear":
            cr = linear_schedule(cr)

        model_params.update(
            {
                "n_steps": int(optuna_params.get("n_steps")),
                "batch_size": int(optuna_params.get("batch_size")),
                "gamma": float(optuna_params.get("gamma")),
                "learning_rate": lr,
                "ent_coef": float(optuna_params.get("ent_coef")),
                "clip_range": cr,
                "n_epochs": int(optuna_params.get("n_epochs")),
                "gae_lambda": float(optuna_params.get("gae_lambda")),
                "max_grad_norm": float(optuna_params.get("max_grad_norm")),
                "vf_coef": float(optuna_params.get("vf_coef")),
            }
        )
        if optuna_params.get("target_kl") is not None:
            model_params["target_kl"] = float(optuna_params.get("target_kl"))
    elif "DQN" in model_type:
        train_freq = optuna_params.get("train_freq")
        subsample_steps = optuna_params.get("subsample_steps")
        gradient_steps = compute_gradient_steps(train_freq, subsample_steps)

        model_params.update(
            {
                "gamma": float(optuna_params.get("gamma")),
                "batch_size": int(optuna_params.get("batch_size")),
                "learning_rate": lr,
                "buffer_size": int(optuna_params.get("buffer_size")),
                "train_freq": train_freq,
                "gradient_steps": gradient_steps,
                "exploration_fraction": float(
                    optuna_params.get("exploration_fraction")
                ),
                "exploration_initial_eps": float(
                    optuna_params.get("exploration_initial_eps")
                ),
                "exploration_final_eps": float(
                    optuna_params.get("exploration_final_eps")
                ),
                "target_update_interval": int(
                    optuna_params.get("target_update_interval")
                ),
                "learning_starts": int(optuna_params.get("learning_starts")),
            }
        )
        if "QRDQN" in model_type and optuna_params.get("n_quantiles") is not None:
            policy_kwargs["n_quantiles"] = int(optuna_params["n_quantiles"])
    else:
        raise ValueError(f"Model {model_type} not supported")

    if optuna_params.get("net_arch"):
        policy_kwargs["net_arch"] = get_net_arch(
            model_type, str(optuna_params["net_arch"])
        )
    if optuna_params.get("activation_fn"):
        policy_kwargs["activation_fn"] = get_activation_fn(
            str(optuna_params["activation_fn"])
        )
    if optuna_params.get("optimizer_class"):
        policy_kwargs["optimizer_class"] = get_optimizer_class(
            str(optuna_params["optimizer_class"])
        )
    if optuna_params.get("ortho_init") is not None:
        policy_kwargs["ortho_init"] = bool(optuna_params["ortho_init"])

    model_params["policy_kwargs"] = policy_kwargs
    return model_params


def sample_params_ppo(trial: Trial, n_envs: int) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams
    """
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    if batch_size > n_steps:
        raise TrialPruned("batch_size must be less than or equal to n_steps")
    if (n_steps * n_envs) % batch_size != 0:
        raise TrialPruned("(n_steps * n_envs) not divisible by batch_size")
    return convert_optuna_params_to_model_params(
        "PPO",
        {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": trial.suggest_categorical(
                "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 0.0005, 0.03, log=True),
            "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            "n_epochs": trial.suggest_categorical("n_epochs", [1, 2, 3, 4, 5]),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.98, step=0.01),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0, step=0.05),
            "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0, step=0.05),
            "lr_schedule": trial.suggest_categorical(
                "lr_schedule", ["linear", "constant"]
            ),
            "cr_schedule": trial.suggest_categorical(
                "cr_schedule", ["linear", "constant"]
            ),
            "target_kl": trial.suggest_categorical(
                "target_kl", [None, 0.01, 0.015, 0.02, 0.03, 0.04]
            ),
            "ortho_init": trial.suggest_categorical("ortho_init", [False, True]),
            "net_arch": trial.suggest_categorical(
                "net_arch", ["small", "medium", "large", "extra_large"]
            ),
            "activation_fn": trial.suggest_categorical(
                "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
            ),
            "optimizer_class": trial.suggest_categorical("optimizer_class", ["adamw"]),
        },
    )


def get_common_dqn_optuna_params(trial: Trial) -> Dict[str, Any]:
    exploration_final_eps = trial.suggest_float(
        "exploration_final_eps", 0.01, 0.2, step=0.01
    )
    exploration_initial_eps = trial.suggest_float(
        "exploration_initial_eps", exploration_final_eps, 1.0
    )
    if exploration_initial_eps >= 0.9:
        min_fraction = 0.2
    elif (exploration_initial_eps - exploration_final_eps) > 0.5:
        min_fraction = 0.15
    else:
        min_fraction = 0.05
    exploration_fraction = trial.suggest_float(
        "exploration_fraction", min_fraction, 0.9, step=0.02
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(5e4), int(1e5), int(2e5)]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
    )
    if learning_starts >= buffer_size:
        raise TrialPruned("learning_starts must be less than buffer_size")
    return {
        "train_freq": trial.suggest_categorical(
            "train_freq", [2, 4, 8, 16, 128, 256, 512, 1024]
        ),
        "subsample_steps": trial.suggest_categorical("subsample_steps", [2, 4, 8]),
        "gamma": trial.suggest_categorical(
            "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [64, 128, 256, 512, 1024]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["linear", "constant"]),
        "buffer_size": buffer_size,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": exploration_fraction,
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [1000, 2000, 5000, 7500, 10000]
        ),
        "learning_starts": learning_starts,
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "medium", "large", "extra_large"]
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
        ),
        "optimizer_class": trial.suggest_categorical("optimizer_class", ["adam"]),
    }


def sample_params_dqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams
    """
    return convert_optuna_params_to_model_params(
        "DQN", get_common_dqn_optuna_params(trial)
    )


def sample_params_qrdqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for QRDQN hyperparams
    """
    dqn_optuna_params = get_common_dqn_optuna_params(trial)
    dqn_optuna_params.update({"n_quantiles": trial.suggest_int("n_quantiles", 10, 160)})
    return convert_optuna_params_to_model_params("QRDQN", dqn_optuna_params)
