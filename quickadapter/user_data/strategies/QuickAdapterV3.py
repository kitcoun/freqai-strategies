import datetime
import json
import logging
import math
from functools import cached_property, lru_cache, reduce
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_absolute
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, isna
from technical.pivots_points import pivots_points

from Utils import (
    TrendDirection,
    alligator,
    bottom_change_percent,
    calculate_n_extrema,
    calculate_quantile,
    ewo,
    format_number,
    get_distance,
    get_zl_ma_fn,
    non_zero_diff,
    price_retracement_percent,
    smooth_extrema,
    top_change_percent,
    vwapb,
    zigzag,
    zlema,
)

debug = False

logger = logging.getLogger(__name__)

EXTREMA_COLUMN = "&s-extrema"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"


class QuickAdapterV3(IStrategy):
    """
    The following freqtrade strategy is released to sponsors of the non-profit FreqAI open-source project.
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

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "3.3.147"

    timeframe = "5m"

    stoploss = -0.02
    use_custom_stoploss = True

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "limit",
        "force_exit": "limit",
        "force_entry": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    position_adjustment_enable = True

    # {stage: (natr_ratio_percent, stake_percent)}
    partial_exit_stages: dict[int, tuple[float, float]] = {
        0: (0.4858, 0.4),
        1: (0.6180, 0.3),
        2: (0.7640, 0.2),
    }

    timeframe_minutes = timeframe_to_minutes(timeframe)
    minimal_roi = {str(timeframe_minutes * 864): -1}

    process_only_new_candles = True

    @cached_property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    @cached_property
    def plot_config(self) -> dict[str, Any]:
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "hp_rmse": {"color": "#c28ce3", "type": "line"},
                    "train_rmse": {"color": "#a3087a", "type": "line"},
                },
                "extrema": {
                    MAXIMA_THRESHOLD_COLUMN: {"color": "#e6be0b", "type": "line"},
                    EXTREMA_COLUMN: {"color": "#f53580", "type": "line"},
                    MINIMA_THRESHOLD_COLUMN: {"color": "#4ae747", "type": "line"},
                },
                "min_max": {
                    "maxima": {"color": "#0dd6de", "type": "bar"},
                    "minima": {"color": "#e3970b", "type": "bar"},
                },
            },
        }

    @cached_property
    def protections(self) -> list[dict[str, Any]]:
        fit_live_predictions_candles = int(
            self.freqai_info.get("fit_live_predictions_candles", 100)
        )
        estimated_trade_duration_candles = int(
            self.config.get("estimated_trade_duration_candles", 36)
        )
        stoploss_guard_lookback_period_candles = int(fit_live_predictions_candles / 2)
        stoploss_guard_trade_limit = max(
            1,
            int(
                round(
                    (
                        stoploss_guard_lookback_period_candles
                        / estimated_trade_duration_candles
                    )
                    * 0.75
                )
            ),
        )
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": 2 * self.config.get("max_open_trades"),
                "stop_duration_candles": fit_live_predictions_candles,
                "max_allowed_drawdown": 0.2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": stoploss_guard_lookback_period_candles,
                "trade_limit": stoploss_guard_trade_limit,
                "stop_duration_candles": stoploss_guard_lookback_period_candles,
                "only_per_pair": True,
            },
        ]

    use_exit_signal = True

    @cached_property
    def startup_candle_count(self) -> int:
        # Match the predictions warmup period
        return self.freqai_info.get("fit_live_predictions_candles", 100)

    @cached_property
    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades")
        if max_open_trades < 0:
            return -1
        if self.is_short_allowed():
            if max_open_trades % 2 == 1:
                max_open_trades += 1
            return int(max_open_trades / 2)
        else:
            return max_open_trades

    def bot_start(self, **kwargs) -> None:
        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI strategy requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or self.freqai_info.get("identifier").strip() == ""
        ):
            raise ValueError(
                "FreqAI strategy requires 'identifier' defined in the freqai section configuration"
            )
        self.models_full_path = Path(
            self.config.get("user_data_dir")
            / "models"
            / self.freqai_info.get("identifier")
        )
        self._label_params: dict[str, dict[str, Any]] = {}
        for pair in self.pairs:
            self._label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": self.freqai_info["feature_parameters"].get(
                        "label_period_candles", 50
                    ),
                    "label_natr_ratio": float(
                        self.freqai_info["feature_parameters"].get(
                            "label_natr_ratio", 6.0
                        )
                    ),
                }
            )
        process_throttle_secs = self.config.get("internals", {}).get(
            "process_throttle_secs", 5
        )
        self._throttle_modulo = max(
            1,
            int(
                round(
                    (timeframe_to_minutes(self.config.get("timeframe")) * 60)
                    / process_throttle_secs
                )
            ),
        )
        self._max_history_size = int(12 * 60 * 60 / process_throttle_secs)
        self._pnl_momentum_window_size = int(20 * 60 / process_throttle_secs)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-aroonosc-period"] = ta.AROONOSC(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(closes, length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-trix-period"] = ta.TRIX(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = pta.cmf(
            highs,
            lows,
            closes,
            volumes,
            length=period,
        )
        dataframe["%-tcp-period"] = top_change_percent(dataframe, period=period)
        dataframe["%-bcp-period"] = bottom_change_percent(dataframe, period=period)
        dataframe["%-prp-period"] = price_retracement_percent(dataframe, period=period)
        dataframe["%-cti-period"] = pta.cti(closes, length=period)
        dataframe["%-chop-period"] = pta.chop(
            highs,
            lows,
            closes,
            length=period,
        )
        dataframe["%-linearreg_angle-period"] = ta.LINEARREG_ANGLE(
            dataframe, timeperiod=period
        )
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        opens = dataframe.get("open")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-close_pct_change"] = closes.pct_change()
        dataframe["%-raw_volume"] = volumes
        dataframe["%-obv"] = ta.OBV(dataframe)
        label_period_candles = self.get_label_period_candles(str(metadata.get("pair")))
        dataframe["%-atr_label_period_candles"] = ta.ATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-ewo"] = ewo(
            dataframe=dataframe,
            pricemode="close",
            mamode="ema",
            zero_lag=True,
            normalize=True,
        )
        psar = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        dataframe["%-diff_to_psar"] = closes - psar
        kc = pta.kc(
            highs,
            lows,
            closes,
            length=14,
            scalar=2,
        )
        dataframe["kc_lowerband"] = kc["KCLe_14_2.0"]
        dataframe["kc_middleband"] = kc["KCBe_14_2.0"]
        dataframe["kc_upperband"] = kc["KCUe_14_2.0"]
        dataframe["%-kc_width"] = (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        ) / dataframe["kc_middleband"]
        (
            dataframe["bb_upperband"],
            dataframe["bb_middleband"],
            dataframe["bb_lowerband"],
        ) = ta.BBANDS(
            ta.TYPPRICE(dataframe),
            timeperiod=14,
            nbdevup=2.2,
            nbdevdn=2.2,
        )
        dataframe["%-bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["%-ibs"] = (closes - lows) / non_zero_diff(highs, lows)
        dataframe["jaw"], dataframe["teeth"], dataframe["lips"] = alligator(
            dataframe, pricemode="median", zero_lag=True
        )
        dataframe["%-dist_to_jaw"] = get_distance(closes, dataframe["jaw"])
        dataframe["%-dist_to_teeth"] = get_distance(closes, dataframe["teeth"])
        dataframe["%-dist_to_lips"] = get_distance(closes, dataframe["lips"])
        dataframe["%-spread_jaw_teeth"] = dataframe["jaw"] - dataframe["teeth"]
        dataframe["%-spread_teeth_lips"] = dataframe["teeth"] - dataframe["lips"]
        dataframe["zlema_50"] = zlema(closes, period=50)
        dataframe["zlema_12"] = zlema(closes, period=12)
        dataframe["zlema_26"] = zlema(closes, period=26)
        dataframe["%-distzlema50"] = get_distance(closes, dataframe["zlema_50"])
        dataframe["%-distzlema12"] = get_distance(closes, dataframe["zlema_12"])
        dataframe["%-distzlema26"] = get_distance(closes, dataframe["zlema_26"])
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        dataframe["%-dist_to_macdsignal"] = get_distance(
            dataframe["%-macd"], dataframe["%-macdsignal"]
        )
        dataframe["%-dist_to_zerohist"] = get_distance(0, dataframe["%-macdhist"])
        # VWAP bands
        (
            dataframe["vwap_lowerband"],
            dataframe["vwap_middleband"],
            dataframe["vwap_upperband"],
        ) = vwapb(dataframe, 20, 1.0)
        dataframe["%-vwap_width"] = (
            dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]
        ) / dataframe["vwap_middleband"]
        dataframe["%-dist_to_vwap_upperband"] = get_distance(
            closes, dataframe["vwap_upperband"]
        )
        dataframe["%-dist_to_vwap_middleband"] = get_distance(
            closes, dataframe["vwap_middleband"]
        )
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(
            closes, dataframe["vwap_lowerband"]
        )
        dataframe["%-body"] = closes - opens
        dataframe["%-tail"] = (np.minimum(opens, closes) - lows).clip(lower=0)
        dataframe["%-wick"] = (highs - np.maximum(opens, closes)).clip(lower=0)
        pp = pivots_points(dataframe)
        dataframe["r1"] = pp["r1"]
        dataframe["s1"] = pp["s1"]
        dataframe["r2"] = pp["r2"]
        dataframe["s2"] = pp["s2"]
        dataframe["r3"] = pp["r3"]
        dataframe["s3"] = pp["s3"]
        dataframe["%-dist_to_r1"] = get_distance(closes, dataframe["r1"])
        dataframe["%-dist_to_r2"] = get_distance(closes, dataframe["r2"])
        dataframe["%-dist_to_r3"] = get_distance(closes, dataframe["r3"])
        dataframe["%-dist_to_s1"] = get_distance(closes, dataframe["s1"])
        dataframe["%-dist_to_s2"] = get_distance(closes, dataframe["s2"])
        dataframe["%-dist_to_s3"] = get_distance(closes, dataframe["s3"])
        dataframe["%-raw_close"] = closes
        dataframe["%-raw_open"] = opens
        dataframe["%-raw_low"] = lows
        dataframe["%-raw_high"] = highs
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dates = dataframe.get("date")

        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25
        return dataframe

    def get_label_period_candles(self, pair: str) -> int:
        label_period_candles = self._label_params.get(pair, {}).get(
            "label_period_candles"
        )
        if label_period_candles and isinstance(label_period_candles, int):
            return label_period_candles
        return self.freqai_info["feature_parameters"].get("label_period_candles", 50)

    def set_label_period_candles(self, pair: str, label_period_candles: int) -> None:
        if isinstance(label_period_candles, int):
            self._label_params[pair]["label_period_candles"] = label_period_candles

    def get_label_natr_ratio(self, pair: str) -> float:
        label_natr_ratio = self._label_params.get(pair, {}).get("label_natr_ratio")
        if label_natr_ratio and isinstance(label_natr_ratio, float):
            return label_natr_ratio
        return float(
            self.freqai_info["feature_parameters"].get("label_natr_ratio", 6.0)
        )

    def set_label_natr_ratio(self, pair: str, label_natr_ratio: float) -> None:
        if isinstance(label_natr_ratio, float) and np.isfinite(label_natr_ratio):
            self._label_params[pair]["label_natr_ratio"] = label_natr_ratio

    def get_label_natr_ratio_percent(self, pair: str, percent: float) -> float:
        if not isinstance(percent, float) or not (0.0 <= percent <= 1.0):
            raise ValueError(
                f"Invalid percent value: {percent}. It should be a float between 0 and 1"
            )
        return self.get_label_natr_ratio(pair) * percent

    @staticmethod
    def td_format(
        delta: datetime.timedelta, pattern: str = "{sign}{d}:{h:02d}:{m:02d}:{s:02d}"
    ) -> str:
        negative_duration = delta.total_seconds() < 0
        delta = abs(delta)
        duration = {"d": delta.days}
        duration["h"], remainder = divmod(delta.seconds, 3600)
        duration["m"], duration["s"] = divmod(remainder, 60)
        duration["ms"] = delta.microseconds // 1000
        duration["sign"] = "-" if negative_duration else ""
        try:
            return pattern.format(**duration)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid pattern '{pattern}': {repr(e)}")

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        pair = str(metadata.get("pair"))
        label_period_candles = self.get_label_period_candles(pair)
        label_natr_ratio = self.get_label_natr_ratio(pair)
        pivots_indices, _, pivots_directions, _ = zigzag(
            dataframe,
            natr_period=label_period_candles,
            natr_ratio=label_natr_ratio,
        )
        label_period = datetime.timedelta(
            minutes=len(dataframe) * timeframe_to_minutes(self.config.get("timeframe"))
        )
        dataframe[EXTREMA_COLUMN] = 0
        if len(pivots_indices) == 0:
            logger.warning(
                f"{pair}: no extrema to label (label_period={QuickAdapterV3.td_format(label_period)} / {label_period_candles=} / {label_natr_ratio=:.2f})"
            )
        else:
            for pivot_idx, pivot_dir in zip(pivots_indices, pivots_directions):
                dataframe.at[pivot_idx, EXTREMA_COLUMN] = pivot_dir
            dataframe["minima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.DOWN, -1, 0
            )
            dataframe["maxima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.UP, 1, 0
            )
            logger.info(
                f"{pair}: labeled {len(pivots_indices)} extrema (label_period={QuickAdapterV3.td_format(label_period)} / {label_period_candles=} / {label_natr_ratio=:.2f})"
            )
        dataframe[EXTREMA_COLUMN] = smooth_extrema(
            dataframe[EXTREMA_COLUMN],
            str(self.freqai_info.get("extrema_smoothing", "gaussian")),
            int(self.freqai_info.get("extrema_smoothing_window", 5)),
            float(self.freqai_info.get("extrema_smoothing_beta", 8.0)),
        )
        if debug:
            extrema = dataframe[EXTREMA_COLUMN]
            logger.info(f"{extrema.to_numpy()=}")
            n_extrema: int = calculate_n_extrema(extrema)
            logger.info(f"{n_extrema=}")
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe.get("DI_values") > dataframe.get("DI_cutoff"),
            0,
            1,
        )

        pair = str(metadata.get("pair"))

        self.set_label_period_candles(
            pair, dataframe.get("label_period_candles").iloc[-1]
        )
        self.set_label_natr_ratio(pair, dataframe.get("label_natr_ratio").iloc[-1])

        dataframe["natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=self.get_label_period_candles(pair)
        )

        dataframe["minima_threshold"] = dataframe.get(MINIMA_THRESHOLD_COLUMN)
        dataframe["maxima_threshold"] = dataframe.get(MAXIMA_THRESHOLD_COLUMN)

        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        enter_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) < dataframe.get("minima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) > dataframe.get("maxima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        return dataframe

    def get_trade_entry_date(self, trade: Trade) -> datetime.datetime:
        return timeframe_to_prev_date(self.config.get("timeframe"), trade.open_date_utc)

    def get_trade_duration_candles(self, df: DataFrame, trade: Trade) -> Optional[int]:
        """
        Get the number of candles since the trade entry.
        :param df: DataFrame with the current data
        :param trade: Trade object
        :return: Number of candles since the trade entry
        """
        entry_date = self.get_trade_entry_date(trade)
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        current_date = dates.iloc[-1]
        if isna(current_date):
            return None
        trade_duration_minutes = (current_date - entry_date).total_seconds() / 60.0
        return int(
            trade_duration_minutes / timeframe_to_minutes(self.config.get("timeframe"))
        )

    @staticmethod
    def is_trade_duration_valid(trade_duration: Optional[int | float]) -> bool:
        return isinstance(trade_duration, (int, float)) and not (
            isna(trade_duration) or trade_duration <= 0
        )

    def get_trade_weighted_interpolation_natr(
        self, df: DataFrame, trade: Trade
    ) -> Optional[float]:
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        entry_date = self.get_trade_entry_date(trade)
        trade_label_natr = label_natr[dates >= entry_date]
        if trade_label_natr.empty:
            return None
        entry_natr = trade_label_natr.iloc[0]
        if isna(entry_natr) or entry_natr < 0:
            return None
        if len(trade_label_natr) == 1:
            return entry_natr
        current_natr = trade_label_natr.iloc[-1]
        if isna(current_natr) or current_natr < 0:
            return None
        median_natr = trade_label_natr.median()

        np_trade_label_natr = trade_label_natr.to_numpy()
        entry_quantile = calculate_quantile(np_trade_label_natr, entry_natr)
        current_quantile = calculate_quantile(np_trade_label_natr, current_natr)
        median_quantile = calculate_quantile(np_trade_label_natr, median_natr)

        if isna(entry_quantile) or isna(current_quantile) or isna(median_quantile):
            return None

        def calculate_weight(
            quantile: float,
            min_weight: float = 0.0,
            max_weight: float = 1.0,
            weighting_exponent: float = 1.5,
        ) -> float:
            normalized_distance_from_center = abs(quantile - 0.5) * 2.0
            return (
                min_weight
                + (max_weight - min_weight)
                * normalized_distance_from_center**weighting_exponent
            )

        entry_weight = calculate_weight(entry_quantile)
        current_weight = calculate_weight(current_quantile)
        median_weight = calculate_weight(median_quantile)

        total_weight = entry_weight + current_weight + median_weight
        if np.isclose(total_weight, 0):
            return None
        entry_weight /= total_weight
        current_weight /= total_weight
        median_weight /= total_weight

        return (
            entry_natr * entry_weight
            + current_natr * current_weight
            + median_natr * median_weight
        )

    def get_trade_interpolation_natr(
        self, df: DataFrame, trade: Trade
    ) -> Optional[float]:
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        entry_date = self.get_trade_entry_date(trade)
        trade_label_natr = label_natr[dates >= entry_date]
        if trade_label_natr.empty:
            return None
        entry_natr = trade_label_natr.iloc[0]
        if isna(entry_natr) or entry_natr < 0:
            return None
        if len(trade_label_natr) == 1:
            return entry_natr
        current_natr = trade_label_natr.iloc[-1]
        if isna(current_natr) or current_natr < 0:
            return None
        trade_volatility_quantile = calculate_quantile(
            trade_label_natr.to_numpy(), entry_natr
        )
        if isna(trade_volatility_quantile):
            return None
        return np.interp(
            trade_volatility_quantile,
            [0.0, 1.0],
            [current_natr, entry_natr],
        )

    def get_trade_moving_average_natr(
        self, df: DataFrame, pair: str, trade_duration_candles: int
    ) -> Optional[float]:
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        if trade_duration_candles >= 2:
            zl_kama = get_zl_ma_fn("kama")
            try:
                trade_kama_natr_values = zl_kama(
                    label_natr, timeperiod=trade_duration_candles
                )
                trade_kama_natr_values = trade_kama_natr_values[
                    ~np.isnan(trade_kama_natr_values)
                ]
                if trade_kama_natr_values.size > 0:
                    return trade_kama_natr_values[-1]
            except Exception as e:
                logger.warning(
                    f"Failed to calculate trade NATR KAMA for pair {pair}: {repr(e)}. Falling back to last trade NATR value",
                    exc_info=True,
                )
        return label_natr.iloc[-1]

    def get_trade_natr(
        self, df: DataFrame, trade: Trade, trade_duration_candles: int
    ) -> Optional[float]:
        trade_price_target = self.config.get("exit_pricing", {}).get(
            "trade_price_target", "moving_average"
        )
        if trade_price_target == "interpolation":
            return self.get_trade_interpolation_natr(df, trade)
        elif trade_price_target == "weighted_interpolation":
            return self.get_trade_weighted_interpolation_natr(df, trade)
        elif trade_price_target == "moving_average":
            return self.get_trade_moving_average_natr(
                df, trade.pair, trade_duration_candles
            )
        else:
            raise ValueError(
                f"Invalid trade_price_target: {trade_price_target}. Expected 'interpolation', 'weighted_interpolation' or 'moving_average'."
            )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_stoploss_log_factor(trade_duration_candles: int) -> float:
        return 1 / math.log10(3.75 + 0.25 * trade_duration_candles)

    def get_stoploss_distance(
        self,
        df: DataFrame,
        trade: Trade,
        current_rate: float,
        natr_ratio_percent: float,
    ) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            current_rate
            * (trade_natr / 100.0)
            * self.get_label_natr_ratio_percent(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_stoploss_log_factor(
                trade_duration_candles
                + int(round(trade.nr_of_successful_exits ** (1.5)))
            )
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_take_profit_log_factor(trade_duration_candles: int) -> float:
        return math.log10(9.75 + 0.25 * trade_duration_candles)

    def get_take_profit_distance(
        self, df: DataFrame, trade: Trade, natr_ratio_percent: float
    ) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            trade.open_rate
            * (trade_natr / 100.0)
            * self.get_label_natr_ratio_percent(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_take_profit_log_factor(trade_duration_candles)
        )

    def throttle_callback(
        self,
        pair: str,
        current_time: datetime.datetime,
        callback: Callable[[], None],
    ) -> None:
        if hash(pair + str(current_time)) % self._throttle_modulo == 0:
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"Error executing callback for {pair}: {repr(e)}", exc_info=True
                )

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        stoploss_distance = self.get_stoploss_distance(df, trade, current_rate, 0.8860)
        if isna(stoploss_distance) or stoploss_distance <= 0:
            return None
        return stoploss_from_absolute(
            current_rate + (1 if trade.is_short else -1) * stoploss_distance,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    @staticmethod
    def can_take_profit(
        trade: Trade, current_rate: float, take_profit_price: float
    ) -> bool:
        return (trade.is_short and current_rate <= take_profit_price) or (
            not trade.is_short and current_rate >= take_profit_price
        )

    def get_take_profit_price(
        self, df: DataFrame, trade: Trade, exit_stage: int
    ) -> Optional[float]:
        natr_ratio_percent = (
            self.partial_exit_stages[exit_stage][0]
            if exit_stage in self.partial_exit_stages
            else 1.0
        )
        take_profit_distance = self.get_take_profit_distance(
            df, trade, natr_ratio_percent
        )
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None

        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )
        trade_take_profit_price_history = (
            QuickAdapterV3.get_trade_take_profit_price_history(trade)
        )
        previous_take_profit_price = (
            trade_take_profit_price_history[-1]
            if trade_take_profit_price_history
            else None
        )
        if previous_take_profit_price != take_profit_price:
            trade_take_profit_price_history = self.append_trade_take_profit_price(
                trade, take_profit_price
            )

        if exit_stage not in self.partial_exit_stages:
            if not trade_take_profit_price_history:
                return None
            return (
                min(trade_take_profit_price_history)
                if trade.is_short
                else max(trade_take_profit_price_history)
            )

        return take_profit_price

    @staticmethod
    def _get_trade_history(trade: Trade) -> dict[str, list[float]]:
        return trade.get_custom_data(
            "history", {"unrealized_pnl": [], "take_profit_price": []}
        )

    @staticmethod
    def get_trade_unrealized_pnl_history(trade: Trade) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        return history.get("unrealized_pnl", [])

    @staticmethod
    def get_trade_take_profit_price_history(trade: Trade) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        return history.get("take_profit_price", [])

    def append_trade_unrealized_pnl(self, trade: Trade, pnl: float) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        pnl_history = history.setdefault("unrealized_pnl", [])
        pnl_history.append(pnl)
        if len(pnl_history) > self._max_history_size:
            pnl_history = pnl_history[-self._max_history_size :]
            history["unrealized_pnl"] = pnl_history
        trade.set_custom_data("history", history)
        return pnl_history

    def append_trade_take_profit_price(
        self, trade: Trade, take_profit_price: float
    ) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        price_history = history.setdefault("take_profit_price", [])
        price_history.append(take_profit_price)
        if len(price_history) > self._max_history_size:
            price_history = price_history[-self._max_history_size :]
            history["take_profit_price"] = price_history
        trade.set_custom_data("history", history)
        return price_history

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float] | tuple[Optional[float], Optional[str]]:
        if trade.has_open_orders:
            return None

        if trade.nr_of_successful_exits not in self.partial_exit_stages:
            return None

        df, _ = self.dp.get_analyzed_dataframe(
            pair=trade.pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        trade_take_profit_price = self.get_take_profit_price(
            df, trade, trade.nr_of_successful_exits
        )
        if isna(trade_take_profit_price):
            return None

        trade_partial_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )
        if not trade_partial_exit:
            self.throttle_callback(
                pair=trade.pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} {trade.pair} stage {trade.nr_of_successful_exits} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)}"
                ),
            )
        if trade_partial_exit:
            trade_stake_percent = self.partial_exit_stages[
                trade.nr_of_successful_exits
            ][1]
            trade_partial_stake_amount = trade.stake_amount * trade_stake_percent
            return (
                -trade_partial_stake_amount,
                f"take_profit_{trade.trade_direction}_{trade.nr_of_successful_exits}",
            )

        return None

    @staticmethod
    def weighted_close(series: Series) -> float:
        return (series.get("high") + series.get("low") + 2 * series.get("close")) / 4.0

    def _calculate_current_deviation(
        self,
        df: DataFrame,
        pair: str,
        min_natr_ratio_percent: float,
        max_natr_ratio_percent: float,
        interpolation_direction: Literal["direct", "inverse"] = "direct",
        quantile_exponent: float = 1.5,
    ) -> Optional[float]:
        label_natr_values = df.get("natr_label_period_candles").to_numpy()
        label_period_candles = self.get_label_period_candles(pair)
        last_label_natr_value = label_natr_values[-1]
        last_label_natr_value_quantile = calculate_quantile(
            label_natr_values[-label_period_candles:], last_label_natr_value
        )
        if isna(last_label_natr_value_quantile):
            last_label_natr_value_quantile = 0.5
        if interpolation_direction == "direct":
            natr_ratio_percent = (
                min_natr_ratio_percent
                + (max_natr_ratio_percent - min_natr_ratio_percent)
                * last_label_natr_value_quantile**quantile_exponent
            )
        elif interpolation_direction == "inverse":
            natr_ratio_percent = (
                max_natr_ratio_percent
                - (max_natr_ratio_percent - min_natr_ratio_percent)
                * last_label_natr_value_quantile**quantile_exponent
            )
        else:
            raise ValueError(
                f"Invalid interpolation_direction: {interpolation_direction}. Expected 'direct' or 'inverse'"
            )
        return (last_label_natr_value / 100.0) * self.get_label_natr_ratio_percent(
            pair, natr_ratio_percent
        )

    def calculate_current_threshold(self, df: DataFrame, pair: str, side: str) -> float:
        current_deviation = self._calculate_current_deviation(
            df,
            pair,
            min_natr_ratio_percent=0.00985,
            max_natr_ratio_percent=0.0995,
            interpolation_direction="direct",
        )
        if isna(current_deviation):
            return np.inf if side == "short" else -np.inf

        last_candle = df.iloc[-1]
        last_candle_close = last_candle.get("close")
        last_candle_open = last_candle.get("open")
        is_last_candle_bullish = last_candle_close > last_candle_open
        is_last_candle_bearish = last_candle_close < last_candle_open

        if side == "long":
            base_price = (
                QuickAdapterV3.weighted_close(last_candle)
                if is_last_candle_bearish
                else last_candle_close
            )
            return base_price * (1 + current_deviation)
        elif side == "short":
            base_price = (
                QuickAdapterV3.weighted_close(last_candle)
                if is_last_candle_bullish
                else last_candle_close
            )
            return base_price * (1 - current_deviation)

        raise ValueError(f"Invalid side: {side}. Expected 'long' or 'short'")

    def get_trade_pnl_momentum(
        self, trade: Trade
    ) -> tuple[float, float, float, float, float, float, float, float]:
        unrealized_pnl_history = QuickAdapterV3.get_trade_unrealized_pnl_history(trade)

        velocity = np.diff(unrealized_pnl_history)
        velocity_std = np.std(velocity) if velocity.size > 1 else 0.0
        acceleration = np.diff(velocity)
        acceleration_std = np.std(acceleration) if acceleration.size > 1 else 0.0

        mean_velocity = np.mean(velocity) if velocity.size > 0 else 0.0
        mean_acceleration = np.mean(acceleration) if acceleration.size > 0 else 0.0

        recent_unrealized_pnl_history = (
            unrealized_pnl_history[-self._pnl_momentum_window_size :]
            if len(unrealized_pnl_history) > self._pnl_momentum_window_size
            else unrealized_pnl_history
        )

        recent_velocity = np.diff(recent_unrealized_pnl_history)
        recent_velocity_std = (
            np.std(recent_velocity) if recent_velocity.size > 1 else 0.0
        )
        recent_acceleration = np.diff(recent_velocity)
        recent_acceleration_std = (
            np.std(recent_acceleration) if recent_acceleration.size > 1 else 0.0
        )

        recent_mean_velocity = (
            np.mean(recent_velocity) if recent_velocity.size > 0 else 0.0
        )
        recent_mean_acceleration = (
            np.mean(recent_acceleration) if recent_acceleration.size > 0 else 0.0
        )

        return (
            mean_velocity,
            velocity_std,
            mean_acceleration,
            acceleration_std,
            recent_mean_velocity,
            recent_velocity_std,
            recent_mean_acceleration,
            recent_acceleration_std,
        )

    @staticmethod
    def is_isoformat(string: str) -> bool:
        if not isinstance(string, str):
            return False
        try:
            datetime.datetime.fromisoformat(string)
        except (ValueError, TypeError):
            return False
        return True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        trade_unrealized_pnl_history = QuickAdapterV3.get_trade_unrealized_pnl_history(
            trade
        )
        previous_unrealized_pnl = (
            trade_unrealized_pnl_history[-1] if trade_unrealized_pnl_history else None
        )
        if previous_unrealized_pnl != current_profit:
            self.append_trade_unrealized_pnl(trade, current_profit)

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        last_candle = df.iloc[-1]
        if last_candle.get("do_predict") == 2:
            return "model_expired"
        if last_candle.get("DI_catch") == 0:
            last_candle_date = last_candle.get("date")
            last_outlier_date_isoformat = trade.get_custom_data("last_outlier_date")
            last_outlier_date = (
                datetime.datetime.fromisoformat(last_outlier_date_isoformat)
                if QuickAdapterV3.is_isoformat(last_outlier_date_isoformat)
                else None
            )
            if last_outlier_date != last_candle_date:
                n_outliers = trade.get_custom_data("n_outliers", 0)
                n_outliers += 1
                logger.warning(
                    f"{pair}: detected new predictions outlier ({n_outliers=}) on trade {trade.id}"
                )
                trade.set_custom_data("n_outliers", n_outliers)
                trade.set_custom_data("last_outlier_date", last_candle_date.isoformat())

        if (
            trade.trade_direction == "short"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
            and current_rate > self.calculate_current_threshold(df, pair, "long")
        ):
            return "minima_detected_short"
        if (
            trade.trade_direction == "long"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
            and current_rate < self.calculate_current_threshold(df, pair, "short")
        ):
            return "maxima_detected_long"

        if trade.nr_of_successful_exits in self.partial_exit_stages:
            return None

        (
            trade_pnl_velocity,
            trade_pnl_velocity_std,
            trade_pnl_acceleration,
            trade_pnl_acceleration_std,
            trade_recent_pnl_velocity,
            trade_recent_pnl_velocity_std,
            trade_recent_pnl_acceleration,
            trade_recent_pnl_acceleration_std,
        ) = self.get_trade_pnl_momentum(trade)
        trade_pnl_momentum_declining = (
            trade_pnl_acceleration < -trade_pnl_acceleration_std * 0.000025
            and trade_pnl_velocity < -trade_pnl_velocity_std * 0.0025
        )
        trade_recent_pnl_spiking = (
            trade_recent_pnl_acceleration > trade_recent_pnl_acceleration_std * 0.00075
            and trade_recent_pnl_velocity > trade_recent_pnl_velocity_std * 0.075
        )

        trade_take_profit_price = self.get_take_profit_price(
            df, trade, trade.nr_of_successful_exits
        )
        if isna(trade_take_profit_price):
            return None
        trade_take_profit_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )

        trade_exit = trade_take_profit_exit and (
            trade_pnl_momentum_declining or not trade_recent_pnl_spiking
        )
        if not trade_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} {trade.pair} stage {trade.nr_of_successful_exits} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)} | "
                    f"Spiking: {trade_recent_pnl_spiking} "
                    f"(V:{format_number(trade_recent_pnl_velocity)} S:{format_number(trade_recent_pnl_velocity_std)}, "
                    f"A:{format_number(trade_recent_pnl_acceleration)} S:{format_number(trade_recent_pnl_acceleration_std)}) | "
                    f"Declining: {trade_pnl_momentum_declining} "
                    f"(V:{format_number(trade_pnl_velocity)} S:{format_number(trade_pnl_velocity_std)}, "
                    f"A:{format_number(trade_pnl_acceleration)} S:{format_number(trade_pnl_acceleration_std)})"
                ),
            )
        if trade_exit:
            return f"take_profit_{trade.trade_direction}_{trade.nr_of_successful_exits}"

        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime.datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        if Trade.get_open_trade_count() >= self.config.get("max_open_trades"):
            return False
        max_open_trades_per_side = self.max_open_trades_per_side
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_open_trades()
            trades_per_side = sum(
                1 for trade in open_trades if trade.trade_direction == side
            )
            if trades_per_side >= max_open_trades_per_side:
                return False

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return False
        current_threshold = self.calculate_current_threshold(df, pair, side)
        if (side == "long" and rate > current_threshold) or (
            side == "short" and rate < current_threshold
        ):
            return True
        logger.info(
            f"User denied {side} entry for {pair}: rate {format_number(rate)} did not break threshold {format_number(current_threshold)}"
        )
        return False

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode == "margin" or trading_mode == "futures":
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")

    def optuna_load_best_params(
        self, pair: str, namespace: str
    ) -> Optional[dict[str, Any]]:
        best_params_path = Path(
            self.models_full_path
            / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None
