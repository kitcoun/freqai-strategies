import logging
from functools import reduce
import datetime
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.persistence import Trade
from scipy.signal import argrelmin, argrelmax
import numpy as np
import pandas_ta as pta

logger = logging.getLogger(__name__)

EXTREMA_COLUMN = "&s-extrema"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"


class QuickAdapterV3(IStrategy):
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

    timeframe = "5m"

    stoploss = -0.02
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.0099
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "limit",
        "force_exit": "limit",
        "force_entry": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    position_adjustment_enable = False
    max_entry_position_adjustment = 1
    max_dca_multiplier = 2

    timeframe_minutes = timeframe_to_minutes(timeframe)
    minimal_roi = {"0": 0.03, str(timeframe_minutes * 864): -1}

    process_only_new_candles = True

    @property
    def can_short(self):
        return self.is_short_allowed()

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "hp_rmse": {"color": "#c28ce3", "type": "line"},
                    "period_rmse": {"color": "#a3087a", "type": "line"},
                },
                "extrema": {
                    EXTREMA_COLUMN: {"color": "#f53580", "type": "line"},
                    MINIMA_THRESHOLD_COLUMN: {"color": "#4ae747", "type": "line"},
                    MAXIMA_THRESHOLD_COLUMN: {"color": "#e6be0b", "type": "line"},
                },
                "min_max": {
                    "maxima": {"color": "#0dd6de", "type": "bar"},
                    "minima": {"color": "#e3970b", "type": "bar"},
                },
            },
        }

    @property
    def protections(self):
        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles", 100
        )
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": int(fit_live_predictions_candles / 2),
                "trade_limit": 1,
                "stop_duration_candles": int(fit_live_predictions_candles / 2),
                "only_per_pair": True,
            },
        ]

    use_exit_signal = True

    @property
    def startup_candle_count(self):
        return int(self.freqai_info.get("fit_live_predictions_candles", 100) / 2)

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe["close"], length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-trix-period"] = ta.TRIX(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-cti-period"] = pta.cti(dataframe["close"], length=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe["close"], timeperiod=period
        )
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = (
            dataframe["%-atr-period"] / dataframe["close"] * 1000
        )
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)
        # Added
        psar = ta.SAR(
            dataframe["high"], dataframe["low"], acceleration=0.02, maximum=0.2
        )
        dataframe["%-diff_to_psar"] = dataframe["close"] - psar
        kc = qtpylib.keltner_channel(dataframe, window=14, atrs=2)
        dataframe["kc_lowerband"] = kc["lower"]
        dataframe["kc_middleband"] = kc["mid"]
        dataframe["kc_upperband"] = kc["upper"]
        dataframe["%-kc_width"] = (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        ) / dataframe["kc_middleband"]
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=14, stds=2.2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["%-bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["%-ibs"] = (
            (dataframe["close"] - dataframe["low"])
            / (dataframe["high"] - dataframe["low"])
        ).fillna(0.0)
        # dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        # dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        # dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        # dataframe["%-distema50"] = get_distance(dataframe["close"], dataframe["ema_50"])
        # dataframe["%-distema12"] = get_distance(dataframe["close"], dataframe["ema_12"])
        # dataframe["%-distema26"] = get_distance(dataframe["close"], dataframe["ema_26"])
        dataframe["zlema_50"] = pta.zlma(dataframe["close"], length=50, mamode="ema")
        dataframe["zlema_12"] = pta.zlma(dataframe["close"], length=12, mamode="ema")
        dataframe["zlema_26"] = pta.zlma(dataframe["close"], length=26, mamode="ema")
        dataframe["%-distzlema50"] = get_distance(
            dataframe["close"], dataframe["zlema_50"]
        )
        dataframe["%-distzlema12"] = get_distance(
            dataframe["close"], dataframe["zlema_12"]
        )
        dataframe["%-distzlema26"] = get_distance(
            dataframe["close"], dataframe["zlema_26"]
        )
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        dataframe["%-dist_to_macdsignal"] = get_distance(
            dataframe["%-macd"], dataframe["%-macdsignal"]
        )
        dataframe["%-dist_to_zerohist"] = get_distance(0, dataframe["%-macdhist"])
        # VWAP
        (
            dataframe["vwap_lowerband"],
            dataframe["vwap_middleband"],
            dataframe["vwap_upperband"],
        ) = VWAPB(dataframe, 20, 1)
        dataframe["%-vwap_width"] = (
            (dataframe["vwap_upperband"] - dataframe["vwap_lowerband"])
            / dataframe["vwap_middleband"]
        ) * 100
        dataframe["%-dist_to_vwap_upperband"] = get_distance(
            dataframe["close"], dataframe["vwap_upperband"]
        )
        dataframe["%-dist_to_vwap_middleband"] = get_distance(
            dataframe["close"], dataframe["vwap_middleband"]
        )
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(
            dataframe["close"], dataframe["vwap_lowerband"]
        )
        dataframe["%-body"] = dataframe["close"] - dataframe["open"]
        dataframe["%-tail"] = (
            np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]
        ).clip(lower=0)
        dataframe["%-wick"] = (
            dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])
        ).clip(lower=0)
        pp = pivots_points(dataframe)
        dataframe["pivot"] = pp["pivot"]
        dataframe["r1"] = pp["r1"]
        dataframe["s1"] = pp["s1"]
        dataframe["r2"] = pp["r2"]
        dataframe["s2"] = pp["s2"]
        dataframe["r3"] = pp["r3"]
        dataframe["s3"] = pp["s3"]
        dataframe["%-dist_to_r1"] = get_distance(dataframe["close"], dataframe["r1"])
        dataframe["%-dist_to_r2"] = get_distance(dataframe["close"], dataframe["r2"])
        dataframe["%-dist_to_r3"] = get_distance(dataframe["close"], dataframe["r3"])
        dataframe["%-dist_to_s1"] = get_distance(dataframe["close"], dataframe["s1"])
        dataframe["%-dist_to_s2"] = get_distance(dataframe["close"], dataframe["s2"])
        dataframe["%-dist_to_s3"] = get_distance(dataframe["close"], dataframe["s3"])
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def set_freqai_targets(self, dataframe, metadata, **kwargs):
        pair = str(metadata.get("pair"))
        label_period_candles = (
            self.freqai_info["feature_parameters"]
            .get(pair, {})
            .get(
                "label_period_candles",
                self.freqai_info["feature_parameters"]["label_period_candles"],
            )
        )
        min_peaks = argrelmin(
            dataframe["low"].values,
            order=label_period_candles,
        )
        max_peaks = argrelmax(
            dataframe["high"].values,
            order=label_period_candles,
        )
        dataframe[EXTREMA_COLUMN] = 0
        for mp in min_peaks[0]:
            dataframe.at[mp, EXTREMA_COLUMN] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, EXTREMA_COLUMN] = 1
        dataframe["minima"] = np.where(dataframe[EXTREMA_COLUMN] == -1, -1, 0)
        dataframe["maxima"] = np.where(dataframe[EXTREMA_COLUMN] == 1, 1, 0)
        dataframe[EXTREMA_COLUMN] = self.smooth_extrema(dataframe[EXTREMA_COLUMN], 5)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"],
            0,
            1,
        )

        dataframe["minima_threshold"] = dataframe[MINIMA_THRESHOLD_COLUMN]
        dataframe["maxima_threshold"] = dataframe[MAXIMA_THRESHOLD_COLUMN]
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["DI_catch"] == 1,
            df[EXTREMA_COLUMN] < df["minima_threshold"],
        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["DI_catch"] == 1,
            df[EXTREMA_COLUMN] > df["maxima_threshold"],
        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )

        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle["DI_catch"] == 0:
            return "outlier_detected"

        enter_tag = trade.enter_tag
        if (enter_tag == "long" or enter_tag == "short") and last_candle[
            "do_predict"
        ] == 2:
            return "model_expired"
        if (
            enter_tag == "short"
            and last_candle["do_predict"] == 1
            and last_candle[EXTREMA_COLUMN] < last_candle["minima_threshold"]
        ):
            return "minima_detected_short"
        if (
            enter_tag == "long"
            and last_candle["do_predict"] == 1
            and last_candle[EXTREMA_COLUMN] > last_candle["maxima_threshold"]
        ):
            return "maxima_detected_long"

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        max_open_trades_per_side = self.max_open_trades_per_side()
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))
            num_shorts, num_longs = 0, 0
            for trade in open_trades:
                if "short" in trade.enter_tag:
                    num_shorts += 1
                elif "long" in trade.enter_tag:
                    num_longs += 1
            if (side == "long" and num_longs >= max_open_trades_per_side) or (
                side == "short" and num_shorts >= max_open_trades_per_side
            ):
                return False

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        if (side == "long" and rate > last_candle["close"] * (1 + 0.0025)) or (
            side == "short" and rate < last_candle["close"] * (1 - 0.0025)
        ):
            return False

        return True

    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades")
        if max_open_trades < 0:
            return -1
        elif max_open_trades == 0 or max_open_trades == 1:
            return max_open_trades
        elif max_open_trades >= 2:
            if self.is_short_allowed():
                return max_open_trades // 2
            else:
                return max_open_trades

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode == "futures":
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")

    def smooth_extrema(
        self,
        series: Series,
        window: int,
        center: bool = True,
        std: float = 0.5,
    ) -> Series:
        extrema_smoothing = self.freqai_info.get("extrema_smoothing", "gaussian")
        return {
            "gaussian": (
                series.rolling(window=window, win_type="gaussian", center=center).mean(
                    std=std
                )
            ),
            "triang": (
                series.rolling(window=window, win_type="triang", center=center).mean()
            ),
            "ewma": series.ewm(span=window).mean(),
            "zlewma": zlewma(series, length=window),
        }.get(
            extrema_smoothing,
            series.rolling(window=window, win_type="gaussian", center=center).mean(
                std=std
            ),
        )


def top_percent_change(dataframe: DataFrame, length: int) -> Series:
    """
    Percentage change of the current close relative from the range maximum open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The period length to look back
    """
    if length == 0:
        return ((dataframe["close"] - dataframe["open"]) / dataframe["open"]).fillna(
            0.0
        )
    else:
        open_max = dataframe["open"].rolling(length).max()
        return ((dataframe["close"] - open_max) / open_max).fillna(0.0)


def chaikin_mf(df: DataFrame, periods=20) -> Series:
    close = df["close"]
    low = df["low"]
    high = df["high"]
    volume = df["volume"]
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(window=periods).sum() / volume.rolling(window=periods).sum()
    return Series(cmf, name="cmf")


# VWAP bands
def VWAPB(dataframe: DataFrame, window=20, num_of_std=1) -> tuple:
    vwap = qtpylib.rolling_vwap(dataframe, window=window)
    rolling_std = vwap.rolling(window=window).std()
    vwap_low = vwap - (rolling_std * num_of_std)
    vwap_high = vwap + (rolling_std * num_of_std)
    return vwap_low, vwap, vwap_high


def EWO(
    dataframe: DataFrame, sma1_length=5, sma2_length=34, normalize=False
) -> DataFrame:
    sma1 = ta.SMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.SMA(dataframe, timeperiod=sma2_length)
    smadif = sma1 - sma2
    if normalize:
        smadif = ((smadif / dataframe["close"]) * 100).fillna(
            0.0
        )  # Optional normalization
    return smadif


def zlewma(series: Series, length: int) -> Series:
    """
    Calculate the ZLEWMA (Zero Lag Exponential Weighted Moving Average) of a series.
    """
    lag = int(round((length - 1) / 2))
    series = series + (series - series.shift(lag))
    return series.ewm(span=length).mean()


def get_distance(p1: Series | float, p2: Series | float) -> Series | float:
    return abs((p1) - (p2))
