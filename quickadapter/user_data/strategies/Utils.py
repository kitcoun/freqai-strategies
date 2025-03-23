import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from technical import qtpylib


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
        ma_fn = lambda df, timeperiod: pta.zlma(
            df["close"], length=timeperiod, mamode=mamode
        )
    else:
        ma_fn = mamodes.get(mamode, mamodes["sma"])
    return ma_fn


def ewo(
    dataframe: pd.DataFrame,
    ma1_length=5,
    ma2_length=34,
    mamode="sma",
    zero_lag=False,
    normalize=False,
) -> pd.Series:
    ma_fn = get_ma_fn(mamode, zero_lag=zero_lag)
    ma1 = ma_fn(dataframe, timeperiod=ma1_length)
    ma2 = ma_fn(dataframe, timeperiod=ma2_length)
    madiff = ma1 - ma2
    if normalize:
        madiff = (
            madiff / dataframe["close"]
        ) * 100  # Optional normalization with close price
    return madiff


def smma(
    df: pd.DataFrame, period: int, mamode="sma", zero_lag=False, offset=0
) -> pd.Series:
    """
    SMoothed Moving Average (SMMA).
    """
    close = df["close"]
    if len(close) < period:
        return pd.Series(index=close.index, dtype=float)

    smma = close.copy()
    smma[: period - 1] = np.nan
    ma_fn = get_ma_fn(mamode, zero_lag=zero_lag)
    smma.iloc[period - 1] = ma_fn(close[:period], timeperiod=period).iloc[-1]

    for i in range(period, len(close)):
        smma.iat[i] = ((period - 1) * smma.iat[i - 1] + smma.iat[i]) / period

    if offset != 0:
        smma = smma.shift(offset)

    return smma


def alligator(
    df: pd.DataFrame,
    jaw_period=13,
    teeth_period=8,
    lips_period=5,
    jaw_shift=8,
    teeth_shift=5,
    lips_shift=3,
    mamode="sma",
    zero_lag=False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bill Williams' Alligator indicator lines.
    """
    median_price = (df["high"] + df["low"]) / 2

    jaw = smma(median_price, period=jaw_period, mamode=mamode, zero_lag=zero_lag).shift(
        jaw_shift
    )
    teeth = smma(
        median_price, period=teeth_period, mamode=mamode, zero_lag=zero_lag
    ).shift(teeth_shift)
    lips = smma(
        median_price, period=lips_period, mamode=mamode, zero_lag=zero_lag
    ).shift(lips_shift)

    return jaw, teeth, lips


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


def frama(prices: pd.Series, period: int = 16, normalize: bool = False) -> pd.Series:
    """
    Calculate FRAMA with optional normalization.

    Args:
        prices: Pandas Series of closing prices.
        period: Lookback window (default=16).
        normalize: Enable range normalization (default=False).

    Returns:
        FRAMA values as a Pandas Series.
    """
    if period % 2 != 0:
        raise ValueError("FRAMA period must be even")

    frama = np.full(len(prices), np.nan)

    for i in range(period - 1, len(prices)):
        prices_array = prices.iloc[i - period + 1 : i + 1].to_numpy()
        D = fractal_dimension(prices_array, period, normalize)
        alpha = np.exp(-4.6 * (D - 1))

        if np.isnan(frama[i - 1]):
            frama[i] = prices_array[-1]
        else:
            frama[i] = alpha * prices_array[-1] + (1 - alpha) * frama[i - 1]

    return pd.Series(frama, index=prices.index)
