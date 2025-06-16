from enum import IntEnum
from functools import lru_cache
import math
from statistics import median
import numpy as np
import pandas as pd
import scipy as sp
import talib.abstract as ta
from typing import Callable, Literal, Union
from technical import qtpylib


def get_distance(
    p1: Union[pd.Series, float], p2: Union[pd.Series, float]
) -> Union[pd.Series, float]:
    return abs(p1 - p2)


def non_zero_diff(s1: pd.Series, s2: pd.Series) -> pd.Series:
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


@lru_cache(maxsize=8)
def _calculate_coeffs(
    window: int,
    win_type: Literal["gaussian", "kaiser", "triang"],
    std: float,
    beta: float,
) -> np.ndarray:
    if win_type == "gaussian":
        coeffs = sp.signal.windows.gaussian(M=window, std=std, sym=True)
    elif win_type == "kaiser":
        coeffs = sp.signal.windows.kaiser(M=window, beta=beta, sym=True)
    elif win_type == "triang":
        coeffs = sp.signal.windows.triang(M=window, sym=True)
    else:
        raise ValueError(f"Unknown window type: {win_type}")
    return coeffs / np.sum(coeffs)


def zero_phase(
    series: pd.Series,
    window: int,
    win_type: Literal["gaussian", "kaiser", "triang"],
    std: float,
    beta: float,
) -> pd.Series:
    if len(series) == 0:
        return series
    if len(series) < window:
        raise ValueError("Series length must be greater than or equal to window size")
    series_values = series.to_numpy()
    b = _calculate_coeffs(window=window, win_type=win_type, std=std, beta=beta)
    a = 1.0
    filtered_values = sp.signal.filtfilt(b, a, series_values)
    return pd.Series(filtered_values, index=series.index)


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
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe.get("close") - previous_close_top) / previous_close_top


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
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )

    return (dataframe.get("close") - previous_close_bottom) / previous_close_bottom


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
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )
    previous_close_high = (
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe.get("close") - previous_close_low) / (
        non_zero_diff(previous_close_high, previous_close_low)
    )


# VWAP bands
def vwapb(dataframe: pd.DataFrame, window: int = 20, std_factor: float = 1.0) -> tuple:
    vwap = qtpylib.rolling_vwap(dataframe, window=window)
    rolling_std = vwap.rolling(window=window, min_periods=window).std()
    vwap_low = vwap - (rolling_std * std_factor)
    vwap_high = vwap + (rolling_std * std_factor)
    return vwap_low, vwap, vwap_high


def calculate_zero_lag(series: pd.Series, period: int) -> pd.Series:
    """Applies a zero lag filter to reduce MA lag."""
    lag = max((period - 1) / 2, 0)
    if lag == 0:
        return series
    return 2 * series - series.shift(int(lag))


def get_ma_fn(mamode: str) -> Callable[[pd.Series, int], np.ndarray]:
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


def get_zl_ma_fn(mamode: str) -> Callable[[pd.Series, int], np.ndarray]:
    ma_fn = get_ma_fn(mamode)
    return lambda series, timeperiod: ma_fn(
        calculate_zero_lag(series, timeperiod), timeperiod=timeperiod
    )


def zlema(series: pd.Series, period: int) -> pd.Series:
    """Ehlers' Zero Lag EMA."""
    lag = max((period - 1) / 2, 0)
    alpha = 2 / (period + 1)
    zl_series = 2 * series - series.shift(int(lag))
    return zl_series.ewm(alpha=alpha, adjust=False).mean()


def _fractal_dimension(highs: np.ndarray, lows: np.ndarray, period: int) -> float:
    """Original fractal dimension computation implementation per Ehlers' paper."""
    if period % 2 != 0:
        raise ValueError("period must be even")

    half_period = period // 2

    H1 = np.max(highs[:half_period])
    L1 = np.min(lows[:half_period])

    H2 = np.max(highs[half_period:])
    L2 = np.min(lows[half_period:])

    H3 = np.max(highs)
    L3 = np.min(lows)

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

    n = len(df)

    highs = df.get("high")
    lows = df.get("low")
    closes = df.get("close")

    if zero_lag:
        highs = calculate_zero_lag(highs, period=period)
        lows = calculate_zero_lag(lows, period=period)
        closes = calculate_zero_lag(closes, period=period)

    fd = pd.Series(np.nan, index=closes.index)
    for i in range(period, n):
        window_highs = highs.iloc[i - period : i]
        window_lows = lows.iloc[i - period : i]
        fd.iloc[i] = _fractal_dimension(
            window_highs.to_numpy(), window_lows.to_numpy(), period
        )

    alpha = np.exp(-4.6 * (fd - 1)).clip(0.01, 1)

    frama = pd.Series(np.nan, index=closes.index)
    frama.iloc[period - 1] = closes.iloc[:period].mean()
    for i in range(period, n):
        if pd.isna(frama.iloc[i - 1]) or pd.isna(alpha.iloc[i]):
            continue
        frama.iloc[i] = (
            alpha.iloc[i] * closes.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i - 1]
        )

    return frama


def smma(series: pd.Series, period: int, zero_lag=False, offset=0) -> pd.Series:
    """
    SMoothed Moving Average (SMMA).

    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=173&Name=Moving_Average_-_Smoothed
    """
    if period <= 0:
        raise ValueError("period must be greater than 0")
    n = len(series)
    if n < period:
        return pd.Series(index=series.index, dtype=float)

    if zero_lag:
        series = calculate_zero_lag(series, period=period)
    smma = pd.Series(np.nan, index=series.index)
    smma.iloc[period - 1] = series.iloc[:period].mean()

    for i in range(period, n):
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
        "close": lambda df: df.get("close"),
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
    prices = get_price_fn(pricemode)(dataframe)

    if zero_lag:
        if mamode == "ema":
            ma_fn = lambda series, timeperiod: zlema(series, period=timeperiod)
        else:
            ma_fn = get_zl_ma_fn(mamode)
    else:
        ma_fn = get_ma_fn(mamode)

    ma1 = ma_fn(prices, timeperiod=ma1_length)
    ma2 = ma_fn(prices, timeperiod=ma2_length)
    madiff = ma1 - ma2
    if normalize:
        madiff = (madiff / prices) * 100.0
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
    prices = get_price_fn(pricemode)(df)

    jaw = smma(prices, period=jaw_period, zero_lag=zero_lag, offset=jaw_shift)
    teeth = smma(prices, period=teeth_period, zero_lag=zero_lag, offset=teeth_shift)
    lips = smma(prices, period=lips_period, zero_lag=zero_lag, offset=lips_shift)

    return jaw, teeth, lips


def find_fractals(df: pd.DataFrame, period: int = 2) -> tuple[list[int], list[int]]:
    n = len(df)
    if n < 2 * period + 1:
        return [], []

    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()

    indices = df.index.tolist()

    fractal_highs = []
    fractal_lows = []

    for i in range(period, n - period):
        is_high_fractal = all(
            highs[i] > highs[i - j] and highs[i] > highs[i + j]
            for j in range(1, period + 1)
        )
        is_low_fractal = all(
            lows[i] < lows[i - j] and lows[i] < lows[i + j]
            for j in range(1, period + 1)
        )

        if is_high_fractal:
            fractal_highs.append(indices[i])
        if is_low_fractal:
            fractal_lows.append(indices[i])

    return fractal_highs, fractal_lows


def calculate_quantile(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return np.nan

    first_value = values[0]
    if np.all(np.isclose(values, first_value)):
        return (
            0.5
            if np.isclose(value, first_value)
            else (0.0 if value < first_value else 1.0)
        )

    return np.sum(values <= value) / values.size


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_ratio: float = 6.0,
) -> tuple[list[int], list[float], list[int], list[float]]:
    n = len(df)
    if df.empty or n < natr_period:
        return [], [], [], []

    natr_values = (ta.NATR(df, timeperiod=natr_period).bfill() / 100.0).to_numpy()

    indices = df.index.tolist()
    thresholds = natr_values * natr_ratio
    closes = df.get("close").to_numpy()
    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()

    state: TrendDirection = TrendDirection.NEUTRAL

    pivots_indices: list[int] = []
    pivots_values: list[float] = []
    pivots_directions: list[int] = []
    pivots_scaled_natrs: list[float] = []
    last_pivot_pos: int = -1

    candidate_pivot_pos: int = -1
    candidate_pivot_value: float = np.nan

    volatility_quantile_cache: dict[int, float] = {}

    def calculate_volatility_quantile(pos: int) -> float:
        if pos not in volatility_quantile_cache:
            start = max(0, pos + 1 - natr_period)
            end = min(pos + 1, n)
            if start >= end:
                volatility_quantile_cache[pos] = np.nan
            else:
                volatility_quantile_cache[pos] = calculate_quantile(
                    natr_values[start:end], natr_values[pos]
                )

        return volatility_quantile_cache[pos]

    def calculate_slope_confirmation_window(
        pos: int,
        min_window: int = 3,
        max_window: int = 5,
    ) -> int:
        volatility_quantile = calculate_volatility_quantile(pos)
        if np.isnan(volatility_quantile):
            return int(round(median([min_window, max_window])))

        return np.clip(
            round(min_window + (max_window - min_window) * volatility_quantile),
            min_window,
            max_window,
        ).astype(int)

    def calculate_slopes_ok_threshold(
        pos: int,
        min_threshold: float = 0.65,
        max_threshold: float = 0.75,
    ) -> float:
        volatility_quantile = calculate_volatility_quantile(pos)
        if np.isnan(volatility_quantile):
            return median([min_threshold, max_threshold])

        return min_threshold + (max_threshold - min_threshold) * volatility_quantile

    @lru_cache(maxsize=4096)
    def calculate_slopes_ok_min_max(slopes_ok_threshold: float) -> tuple[int, int]:
        raw_bound1 = math.ceil(1 / slopes_ok_threshold)
        raw_bound2 = math.ceil(1 / (1 - slopes_ok_threshold))
        return min(raw_bound1, raw_bound2), max(raw_bound1, raw_bound2)

    def calculate_min_slopes_ok(pos: int, slopes_ok_threshold: float) -> int:
        min_slopes_ok, max_slopes_ok = calculate_slopes_ok_min_max(slopes_ok_threshold)
        volatility_quantile = calculate_volatility_quantile(pos)
        if np.isnan(volatility_quantile):
            return int(round(median([min_slopes_ok, max_slopes_ok])))

        return np.clip(
            round(
                min_slopes_ok + (max_slopes_ok - min_slopes_ok) * volatility_quantile
            ),
            min_slopes_ok,
            max_slopes_ok,
        ).astype(int)

    def update_candidate_pivot(pos: int, value: float):
        nonlocal candidate_pivot_pos, candidate_pivot_value
        if 0 <= pos < n:
            candidate_pivot_pos = pos
            candidate_pivot_value = value

    def reset_candidate_pivot():
        nonlocal candidate_pivot_pos, candidate_pivot_value
        candidate_pivot_pos = -1
        candidate_pivot_value = np.nan

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)
        pivots_scaled_natrs.append(thresholds[pos])
        last_pivot_pos = pos
        reset_candidate_pivot()

    slope_ok_cache: dict[tuple[int, int, bool, int, float], bool] = {}

    def get_slope_ok(
        pos: int,
        direction: TrendDirection,
        enable_weighting: bool,
        slope_confirmation_window: int,
        min_slope: float,
    ) -> bool:
        cache_key = (
            pos,
            direction.value,
            enable_weighting,
            slope_confirmation_window,
            min_slope,
        )

        if cache_key in slope_ok_cache:
            return slope_ok_cache[cache_key]

        next_start = pos
        next_end = min(next_start + slope_confirmation_window, n)
        next_closes = closes[next_start:next_end]

        if len(next_closes) < 2:
            slope_ok_cache[cache_key] = False
            return slope_ok_cache[cache_key]

        log_next_closes = np.log(next_closes)
        log_next_closes_length = len(log_next_closes)

        polyfit_kwargs = {}
        if enable_weighting:
            polyfit_kwargs["w"] = np.linspace(0.5, 1.5, log_next_closes_length)
        log_next_slope = np.polyfit(
            range(log_next_closes_length),
            log_next_closes,
            1,
            **polyfit_kwargs,
        )[0]

        if direction == TrendDirection.DOWN:
            slope_ok_cache[cache_key] = log_next_slope < -min_slope
        elif direction == TrendDirection.UP:
            slope_ok_cache[cache_key] = log_next_slope > min_slope
        else:
            slope_ok_cache[cache_key] = False

        return slope_ok_cache[cache_key]

    def is_pivot_confirmed(
        pos: int,
        candidate_pivot_pos: int,
        direction: TrendDirection,
        enable_weighting: bool = False,
        min_slope: float = np.finfo(float).eps,
    ) -> bool:
        slope_confirmation_window = calculate_slope_confirmation_window(
            candidate_pivot_pos
        )

        slopes_ok: list[bool] = []
        for i in range(candidate_pivot_pos + 1, min(pos + 1, n)):
            slopes_ok.append(
                get_slope_ok(
                    pos=i,
                    direction=direction,
                    enable_weighting=enable_weighting,
                    slope_confirmation_window=slope_confirmation_window,
                    min_slope=min_slope,
                )
            )

        slopes_ok_threshold = calculate_slopes_ok_threshold(candidate_pivot_pos)
        min_slopes_ok = calculate_min_slopes_ok(
            candidate_pivot_pos, slopes_ok_threshold
        )
        n_slopes_ok = len(slopes_ok)
        if n_slopes_ok > 0:
            return (
                n_slopes_ok >= min_slopes_ok
                and sum(slopes_ok) / n_slopes_ok >= slopes_ok_threshold
            )

        return False

    start_pos = 0
    initial_high_pos = start_pos
    initial_low_pos = start_pos
    initial_high = highs[initial_high_pos]
    initial_low = lows[initial_low_pos]
    for i in range(start_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]
        if current_high > initial_high:
            initial_high, initial_high_pos = current_high, i
        if current_low < initial_low:
            initial_low, initial_low_pos = current_low, i

        initial_move_from_high = (initial_high - current_low) / initial_high
        initial_move_from_low = (current_high - initial_low) / initial_low
        is_initial_high_move_significant = (
            initial_move_from_high >= thresholds[initial_high_pos]
        )
        is_initial_low_move_significant = (
            initial_move_from_low >= thresholds[initial_low_pos]
        )
        if is_initial_high_move_significant and is_initial_low_move_significant:
            if initial_move_from_high > initial_move_from_low:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            else:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
        else:
            if is_initial_high_move_significant:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            elif is_initial_low_move_significant:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
    else:
        return [], [], [], []

    for i in range(last_pivot_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]

        if state == TrendDirection.UP:
            if np.isnan(candidate_pivot_value) or current_high > candidate_pivot_value:
                update_candidate_pivot(i, current_high)
            if (
                candidate_pivot_value - current_low
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.DOWN):
                add_pivot(candidate_pivot_pos, candidate_pivot_value, TrendDirection.UP)
                state = TrendDirection.DOWN

        elif state == TrendDirection.DOWN:
            if np.isnan(candidate_pivot_value) or current_low < candidate_pivot_value:
                update_candidate_pivot(i, current_low)
            if (
                current_high - candidate_pivot_value
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.UP):
                add_pivot(
                    candidate_pivot_pos, candidate_pivot_value, TrendDirection.DOWN
                )
                state = TrendDirection.UP

    return pivots_indices, pivots_values, pivots_directions, pivots_scaled_natrs
