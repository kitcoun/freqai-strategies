from enum import IntEnum
import numpy as np
import pandas as pd
import talib.abstract as ta
from typing import Callable, Union
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from technical import qtpylib


def get_distance(
    p1: Union[pd.Series, float], p2: Union[pd.Series, float]
) -> Union[pd.Series, float]:
    return abs(p1 - p2)


def non_zero_range(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Returns the difference of two series and adds epsilon to any zero values."""
    diff = s1 - s2
    diff = diff.mask(diff == 0, other=diff + np.finfo(float).eps)
    return diff


def get_gaussian_window(std: float, center: bool) -> int:
    if std is None:
        raise ValueError("Standard deviation cannot be None")
    if std <= 0:
        raise ValueError("Standard deviation must be greater than 0")
    window = int(6 * std + 1)
    if center and window % 2 == 0:
        window += 1
    return max(3, window)


def get_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("Window size must be greater than 0")
    return window if window % 2 == 1 else window + 1


def derive_gaussian_std_from_window(window: int) -> float:
    # Assuming window = 6 * std + 1 => std = (window - 1) / 6
    return (window - 1) / 6.0 if window > 1 else 0.5


def zero_phase_gaussian(series: pd.Series, window: int, std: float):
    kernel = gaussian(window, std=std)
    kernel /= kernel.sum()

    padding_length = window - 1
    padded_series = np.pad(series.values, (padding_length, padding_length), mode="edge")

    forward = convolve(padded_series, kernel, mode="valid")
    backward = convolve(forward[::-1], kernel, mode="valid")[::-1]

    return pd.Series(backward, index=series.index)


def top_change_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Percentage change of the current close relative to the top close price in the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: The previous period window size to look back (>=1)
    :return: The top change percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_top = (
        dataframe["close"].rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe["close"] - previous_close_top) / previous_close_top


def bottom_change_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Percentage change of the current close relative to the bottom close price in the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: The previous period window size to look back (>=1)
    :return: The bottom change percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_bottom = (
        dataframe["close"].rolling(period, min_periods=period).min().shift(1)
    )

    return (dataframe["close"] - previous_close_bottom) / previous_close_bottom


def price_retracement_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate the percentage retracement of the current close within the high/low close price range
    of the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: Window size for calculating historical closes high/low (>=1)
    :return: Retracement percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_low = (
        dataframe["close"].rolling(period, min_periods=period).min().shift(1)
    )
    previous_close_high = (
        dataframe["close"].rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe["close"] - previous_close_low) / (
        non_zero_range(previous_close_high, previous_close_low)
    )


# VWAP bands
def vwapb(dataframe: pd.DataFrame, window=20, num_of_std=1) -> tuple:
    vwap = qtpylib.rolling_vwap(dataframe, window=window)
    rolling_std = vwap.rolling(window=window, min_periods=window).std()
    vwap_low = vwap - (rolling_std * num_of_std)
    vwap_high = vwap + (rolling_std * num_of_std)
    return vwap_low, vwap, vwap_high


def zero_lag_series(series: pd.Series, period: int) -> pd.Series:
    """Applies a zero lag filter to reduce MA lag."""
    lag = int(0.5 * (period - 1))
    return 2 * series - series.shift(lag)


def get_ma_fn(mamode: str) -> Callable[[pd.Series, int], pd.Series]:
    mamodes: dict = {
        "sma": ta.SMA,
        "ema": ta.EMA,
        "wma": ta.WMA,
        "dema": ta.DEMA,
        "tema": ta.TEMA,
        "trima": ta.TRIMA,
        "kama": ta.KAMA,
        "t3": ta.T3,
    }
    return mamodes.get(mamode, mamodes["sma"])


def _fractal_dimension(high: np.ndarray, low: np.ndarray, period: int) -> float:
    """Original fractal dimension computation implementation per Ehlers' paper."""
    if period % 2 != 0:
        raise ValueError("period must be even")

    half_period = period // 2

    H1 = np.max(high[:half_period])
    L1 = np.min(low[:half_period])

    H2 = np.max(high[half_period:])
    L2 = np.min(low[half_period:])

    H3 = np.max(high)
    L3 = np.min(low)

    HL1 = H1 - L1
    HL2 = H2 - L2
    HL3 = H3 - L3

    if (HL1 + HL2) == 0 or HL3 == 0:
        return 1.0

    D = (np.log(HL1 + HL2) - np.log(HL3)) / np.log(2)
    return np.clip(D, 1.0, 2.0)


def frama(df: pd.DataFrame, period: int = 16, zero_lag=False) -> pd.Series:
    """
    Original FRAMA implementation per Ehlers' paper with optional zero lag.
    """
    if period % 2 != 0:
        raise ValueError("period must be even")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    if zero_lag:
        high = zero_lag_series(high, period=period)
        low = zero_lag_series(low, period=period)
        close = zero_lag_series(close, period=period)

    fd = pd.Series(np.nan, index=close.index)
    for i in range(period, len(close)):
        window_high = high.iloc[i - period : i]
        window_low = low.iloc[i - period : i]
        fd.iloc[i] = _fractal_dimension(window_high.values, window_low.values, period)

    alpha = np.exp(-4.6 * (fd - 1)).clip(0.01, 1)

    frama = pd.Series(np.nan, index=close.index)
    frama.iloc[period - 1] = close.iloc[:period].mean()
    for i in range(period, len(close)):
        if pd.isna(frama.iloc[i - 1]) or pd.isna(alpha.iloc[i]):
            continue
        frama.iloc[i] = (
            alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i - 1]
        )

    return frama


def smma(series: pd.Series, period: int, zero_lag=False, offset=0) -> pd.Series:
    """
    SMoothed Moving Average (SMMA).

    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=173&Name=Moving_Average_-_Smoothed
    """
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if len(series) < period:
        return pd.Series(index=series.index, dtype=float)

    if zero_lag:
        series = zero_lag_series(series, period=period)
    smma = pd.Series(np.nan, index=series.index)
    smma.iloc[period - 1] = series.iloc[:period].mean()

    for i in range(period, len(series)):
        smma.iloc[i] = (smma.iloc[i - 1] * (period - 1) + series.iloc[i]) / period

    if offset != 0:
        smma = smma.shift(offset)

    return smma


def get_price_fn(pricemode: str) -> Callable[[pd.DataFrame], pd.Series]:
    pricemodes = {
        "average": ta.AVGPRICE,
        "median": ta.MEDPRICE,
        "typical": ta.TYPPRICE,
        "weighted-close": ta.WCLPRICE,
        "close": lambda df: df["close"],
    }
    return pricemodes.get(pricemode, pricemodes["close"])


def ewo(
    dataframe: pd.DataFrame,
    ma1_length=5,
    ma2_length=34,
    pricemode="close",
    mamode="sma",
    zero_lag=False,
    normalize=False,
) -> pd.Series:
    """
    Calculate the Elliott Wave Oscillator (EWO) using two moving averages.
    """
    price_series = get_price_fn(pricemode)(dataframe)

    if zero_lag:
        price_series_ma1 = zero_lag_series(price_series, period=ma1_length)
        price_series_ma2 = zero_lag_series(price_series, period=ma2_length)
    else:
        price_series_ma1 = price_series
        price_series_ma2 = price_series

    ma_fn = get_ma_fn(mamode)
    ma1 = ma_fn(price_series_ma1, timeperiod=ma1_length)
    ma2 = ma_fn(price_series_ma2, timeperiod=ma2_length)
    madiff = ma1 - ma2
    if normalize:
        madiff = (madiff / price_series) * 100
    return madiff


def alligator(
    df: pd.DataFrame,
    jaw_period=13,
    teeth_period=8,
    lips_period=5,
    jaw_shift=8,
    teeth_shift=5,
    lips_shift=3,
    pricemode="median",
    zero_lag=False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bill Williams' Alligator indicator lines.
    """
    price_series = get_price_fn(pricemode)(df)

    jaw = smma(price_series, period=jaw_period, zero_lag=zero_lag, offset=jaw_shift)
    teeth = smma(
        price_series, period=teeth_period, zero_lag=zero_lag, offset=teeth_shift
    )
    lips = smma(price_series, period=lips_period, zero_lag=zero_lag, offset=lips_shift)

    return jaw, teeth, lips


def find_fractals(df: pd.DataFrame, fractal_period: int) -> tuple[list[int], list[int]]:
    if len(df) < 2 * fractal_period + 1:
        return [], []

    highs = df["high"].values
    lows = df["low"].values

    fractal_candidate_indices = np.arange(fractal_period, len(df) - fractal_period)

    is_fractal_high = np.ones(len(fractal_candidate_indices), dtype=bool)
    is_fractal_low = np.ones(len(fractal_candidate_indices), dtype=bool)

    for i in range(1, fractal_period + 1):
        is_fractal_high &= (
            highs[fractal_candidate_indices] > highs[fractal_candidate_indices - i]
        ) & (highs[fractal_candidate_indices] > highs[fractal_candidate_indices + i])

        is_fractal_low &= (
            lows[fractal_candidate_indices] < lows[fractal_candidate_indices - i]
        ) & (lows[fractal_candidate_indices] < lows[fractal_candidate_indices + i])

        if not np.any(is_fractal_high) and not np.any(is_fractal_low):
            break

    return (
        fractal_candidate_indices[is_fractal_high].tolist(),
        fractal_candidate_indices[is_fractal_low].tolist(),
    )


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_ratio: float = 1.0,
    confirmation_window: int = 6,
    depth: int = 12,
) -> tuple[list[int], list[float], list[int]]:
    if df.empty or len(df) < natr_period + confirmation_window:
        return [], [], []

    indices = df.index.tolist()
    thresholds = (
        (ta.NATR(df, timeperiod=natr_period) * natr_ratio).fillna(method="bfill").values
    )
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    state: TrendDirection = TrendDirection.NEUTRAL
    pivots_indices, pivots_values, pivots_directions = [], [], []
    last_pivot_pos = -depth - 1

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)
        last_pivot_pos = pos

    def update_last_pivot(pos: int, value: float, direction: TrendDirection):
        if pivots_indices:
            pivots_indices[-1] = indices[pos]
            pivots_values[-1] = value
            pivots_directions[-1] = direction

    def is_reversal_confirmed(pos: int, direction: TrendDirection) -> bool:
        if pos + confirmation_window >= len(df):
            return False
        next_closes = closes[pos + 1 : pos + confirmation_window + 1]
        if direction == TrendDirection.DOWN:
            return np.all(next_closes < highs[pos])
        elif direction == TrendDirection.UP:
            return np.all(next_closes > lows[pos])
        return False

    start_pos = 0
    initial_high_pos = start_pos
    initial_low_pos = start_pos
    initial_high = highs[initial_high_pos]
    initial_low = lows[initial_low_pos]
    for i in range(start_pos + 1, len(df)):
        if highs[i] > initial_high:
            initial_high, initial_high_pos = highs[i], i
        if lows[i] < initial_low:
            initial_low, initial_low_pos = lows[i], i

        initial_move_from_high = (initial_high - lows[i]) / initial_high
        initial_move_from_low = (highs[i] - initial_low) / initial_low
        if initial_move_from_high >= thresholds[i] and is_reversal_confirmed(
            initial_high_pos, TrendDirection.DOWN
        ):
            add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
            state = TrendDirection.DOWN
            break
        elif initial_move_from_low >= thresholds[i] and is_reversal_confirmed(
            initial_low_pos, TrendDirection.UP
        ):
            add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
            state = TrendDirection.UP
            break
    else:
        return [], [], []

    for i in range(last_pivot_pos + 1, len(df)):
        current_high = highs[i]
        current_low = lows[i]
        last_pivot_val = pivots_values[-1]
        if state == TrendDirection.UP:
            if current_high > last_pivot_val:
                update_last_pivot(i, current_high, TrendDirection.UP)
            elif (
                (last_pivot_val - current_low) / last_pivot_val >= thresholds[i]
                and (i - last_pivot_pos) >= depth
                and is_reversal_confirmed(i, TrendDirection.DOWN)
            ):
                add_pivot(i, current_low, TrendDirection.DOWN)
                state = TrendDirection.DOWN
        elif state == TrendDirection.DOWN:
            if current_low < last_pivot_val:
                update_last_pivot(i, current_low, TrendDirection.DOWN)
            elif (
                (current_high - last_pivot_val) / last_pivot_val >= thresholds[i]
                and (i - last_pivot_pos) >= depth
                and is_reversal_confirmed(i, TrendDirection.UP)
            ):
                add_pivot(i, current_high, TrendDirection.UP)
                state = TrendDirection.UP

    return pivots_indices, pivots_values, pivots_directions
