import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from sys import float_info
from technical import qtpylib


def get_distance(p1: pd.Series | float, p2: pd.Series | float) -> pd.Series | float:
    return abs(p1 - p2)


def non_zero_range(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Returns the difference of two series and adds epsilon to any zero values."""
    diff = s1 - s2
    if diff.eq(0).any().any():
        diff += float_info.epsilon
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


def get_ma_fn(mamode: str, zero_lag=False) -> callable:
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
    if zero_lag:
        ma_fn = lambda series, timeperiod: pta.zlma(
            series, length=timeperiod, mamode=mamode
        )
    else:
        ma_fn = mamodes.get(mamode, mamodes["sma"])
    return ma_fn


def fractal_dimension(
    prices_array: np.ndarray, period: int, normalize: bool = False
) -> float:
    """
    Calculate fractal dimension of a price window, with optional normalization.

    Args:
        window: Array of prices for the current window.
        period: Window size (must be even).
        normalize: If True, normalize HL values by their window lengths.

    Returns:
        Fractal dimension (D) clipped between 1.0 and 2.0.
    """
    if period % 2 != 0:
        raise ValueError("FRAMA period must be even")

    half_period = period // 2
    if half_period < 1 or len(prices_array) < period:
        return 1.0
    prices_first_half = prices_array[:half_period]
    prices_second_half = prices_array[half_period:]

    HL1 = np.max(prices_first_half) - np.min(prices_first_half)
    HL2 = np.max(prices_second_half) - np.min(prices_second_half)
    HL3 = np.max(prices_array) - np.min(prices_array)

    if normalize:
        HL1 /= half_period
        HL2 /= half_period
        HL3 /= period

    if HL1 + HL2 == 0 or HL3 == 0:
        return 1.0

    D = (np.log(HL1 + HL2) - np.log(HL3)) / np.log(2)
    return np.clip(D, 1.0, 2.0)


def frama(series: pd.Series, period: int = 16, normalize: bool = False) -> pd.Series:
    """
    Calculate FRAMA with optional normalization.

    Args:
        series: Pandas Series of prices.
        period: Lookback window (default=16).
        normalize: Enable range normalization (default=False).

    Returns:
        FRAMA values as a Pandas Series.
    """
    if period % 2 != 0:
        raise ValueError("FRAMA period must be even")

    frama = np.full(len(series), np.nan)

    for i in range(period - 1, len(series)):
        prices_array = series.iloc[i - period + 1 : i + 1].to_numpy()
        D = fractal_dimension(prices_array, period, normalize)
        alpha = np.exp(-4.6 * (D - 1))

        if np.isnan(frama[i - 1]):
            frama[i] = prices_array[-1]
        else:
            frama[i] = alpha * prices_array[-1] + (1 - alpha) * frama[i - 1]

    return pd.Series(frama, index=series.index)


def smma(
    series: pd.Series, period: int, mamode="sma", zero_lag=False, offset=0
) -> pd.Series:
    """
    SMoothed Moving Average (SMMA).

    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=173&Name=Moving_Average_-_Smoothed
    """
    if len(series) < period:
        return pd.Series(index=series.index, dtype=float)

    smma = series.copy()
    smma[: period - 1] = np.nan
    ma_fn = get_ma_fn(mamode, zero_lag=zero_lag)
    smma.iloc[period - 1] = pd.Series(
        ma_fn(series.iloc[:period], timeperiod=period)
    ).iloc[-1]

    for i in range(period, len(series)):
        smma.iat[i] = ((period - 1) * smma.iat[i - 1] + smma.iat[i]) / period

    if offset != 0:
        smma = smma.shift(offset)

    return smma


def get_price_fn(pricemode: str) -> callable:
    pricemodes = {
        "typical": lambda df: (df["high"] + df["low"] + df["close"]) / 3,
        "median": lambda df: (df["high"] + df["low"]) / 2,
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
    ma_fn = get_ma_fn(mamode, zero_lag=zero_lag)
    ma1 = ma_fn(price_series, timeperiod=ma1_length)
    ma2 = ma_fn(price_series, timeperiod=ma2_length)
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
    mamode="sma",
    zero_lag=False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bill Williams' Alligator indicator lines.
    """
    price_series = get_price_fn(pricemode)(df)

    jaw = smma(price_series, period=jaw_period, mamode=mamode, zero_lag=zero_lag).shift(
        jaw_shift
    )
    teeth = smma(
        price_series, period=teeth_period, mamode=mamode, zero_lag=zero_lag
    ).shift(teeth_shift)
    lips = smma(
        price_series, period=lips_period, mamode=mamode, zero_lag=zero_lag
    ).shift(lips_shift)

    return jaw, teeth, lips
