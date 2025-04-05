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
    rolling_std = vwap.rolling(window=window).std()
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


def __fractal_dimension(high: np.ndarray, low: np.ndarray, period: int) -> float:
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
        fd.iloc[i] = __fractal_dimension(window_high.values, window_low.values, period)

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


def zigzag(
    df: pd.DataFrame,
    threshold: float = 0.05,
) -> tuple[list, list, list]:
    """
    Calculate the ZigZag indicator for a OHLCV DataFrame.

    Parameters:
    df (pd.DataFrame): OHLCV DataFrame.
    threshold (float): Percentage threshold for reversal (default 0.05 for 5%).

    Returns:
    tuple: Lists of indices, extrema, and directions.
    """
    if df.empty:
        return [], [], []

    indices = []
    extrema = []
    directions = []

    current_dir = 0
    current_extreme = None
    current_extreme_idx = None

    first_high = df["high"].iloc[0]
    first_low = df["low"].iloc[0]

    if (first_high - first_low) / first_low >= threshold:
        current_dir = 1
        current_extreme = first_high
    else:
        current_dir = -1
        current_extreme = first_low
    current_extreme_idx = df.index[0]

    indices.append(current_extreme_idx)
    extrema.append(current_extreme)
    directions.append(current_dir)
    last_idx = current_extreme_idx

    for i in range(1, len(df)):
        current_idx = df.index[i]
        h = df.at[current_idx, "high"]
        l = df.at[current_idx, "low"]

        if current_dir == 1:  # Looking for higher high
            if h > current_extreme:
                current_extreme = h
                current_extreme_idx = current_idx
            elif (current_extreme - l) / current_extreme >= threshold:
                if current_extreme_idx != last_idx:
                    indices.append(current_extreme_idx)
                    extrema.append(current_extreme)
                    directions.append(current_dir)
                    last_idx = current_extreme_idx

                current_dir = -1
                current_extreme = l
                current_extreme_idx = current_idx

        elif current_dir == -1:  # Looking for lower low
            if l < current_extreme:
                current_extreme = l
                current_extreme_idx = current_idx
            elif (h - current_extreme) / current_extreme >= threshold:
                if current_extreme_idx != last_idx:
                    indices.append(current_extreme_idx)
                    extrema.append(current_extreme)
                    directions.append(current_dir)
                    last_idx = current_extreme_idx

                current_dir = 1
                current_extreme = h
                current_extreme_idx = current_idx

    if current_extreme_idx != last_idx:
        indices.append(current_extreme_idx)
        extrema.append(current_extreme)
        directions.append(current_dir)

    return indices[1:], extrema[1:], directions[1:]


def dynamic_zigzag(
    df: pd.DataFrame,
    timeperiod: int = 14,
    natr: bool = True,
    ratio: float = 1.0,
) -> tuple[list, list, list]:
    """
    Calculate the ZigZag indicator for a OHLCV DataFrame with dynamic threshold using ATR/NATR.

    Parameters:
    df (pd.DataFrame): OHLCV DataFrame.
    timeperiod (int): Period for ATR/NATR calculation (default: 14).
    natr (bool): Use NATR (True) or ATR (False) (default: True).
    ratio (float): ratio for dynamic threshold (default: 1.0).

    Returns:
    tuple: Lists of indices, extrema, and directions.
    """
    if df.empty:
        return [], [], []

    if natr:
        thresholds = ta.NATR(df, timeperiod=timeperiod)
    else:
        thresholds = ta.ATR(df, timeperiod=timeperiod)
    thresholds = thresholds.ffill().bfill()

    indices = []
    extrema = []
    directions = []

    current_dir = 0
    current_extreme = None
    current_extreme_idx = None

    first_high = df["high"].iloc[0]
    first_low = df["low"].iloc[0]
    first_threshold = thresholds.iloc[0] * ratio

    if natr:
        first_move = (first_high - first_low) / first_low
    else:
        first_move = first_high - first_low
    if first_move >= first_threshold:
        current_dir = 1
        current_extreme = first_high
    else:
        current_dir = -1
        current_extreme = first_low
    current_extreme_idx = df.index[0]

    indices.append(current_extreme_idx)
    extrema.append(current_extreme)
    directions.append(current_dir)
    last_idx = current_extreme_idx

    for i in range(1, len(df)):
        current_idx = df.index[i]
        h = df.at[current_idx, "high"]
        l = df.at[current_idx, "low"]
        threshold = thresholds.iloc[i] * ratio

        if current_dir == 1:  # Looking for higher high
            if h > current_extreme:
                current_extreme = h
                current_extreme_idx = current_idx
            else:
                if natr:
                    reversal = (current_extreme - l) / current_extreme >= threshold
                else:
                    reversal = (current_extreme - l) >= threshold
                if reversal:
                    if current_extreme_idx != last_idx:
                        indices.append(current_extreme_idx)
                        extrema.append(current_extreme)
                        directions.append(current_dir)
                        last_idx = current_extreme_idx

                    current_dir = -1
                    current_extreme = l
                    current_extreme_idx = current_idx

        elif current_dir == -1:  # Looking for lower low
            if l < current_extreme:
                current_extreme = l
                current_extreme_idx = current_idx
            else:
                if natr:
                    reversal = (h - current_extreme) / current_extreme >= threshold
                else:
                    reversal = (h - current_extreme) >= threshold
                if reversal:
                    if current_extreme_idx != last_idx:
                        indices.append(current_extreme_idx)
                        extrema.append(current_extreme)
                        directions.append(current_dir)
                        last_idx = current_extreme_idx

                    current_dir = 1
                    current_extreme = h
                    current_extreme_idx = current_idx

    if current_extreme_idx != last_idx:
        indices.append(current_extreme_idx)
        extrema.append(current_extreme)
        directions.append(current_dir)

    return indices[1:], extrema[1:], directions[1:]
