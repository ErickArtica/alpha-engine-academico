"""Canonical causal V3 feature calculation for offline and closed LIVE klines."""

import numpy as np
import pandas as pd


V3_FEATURE_COLUMNS = (
    "return_1m", "return_3m", "return_5m", "return_15m", "rsi_14",
    "ema_distance", "macd_relative", "macd_signal_relative",
    "macd_histogram_relative", "candle_range", "body_return",
    "upper_wick", "lower_wick", "rolling_volatility", "volume_change",
    "volume_zscore", "trades_zscore", "taker_buy_ratio",
)
V3_WINDOW = 20


def _zscore(values: pd.Series, window: int = V3_WINDOW) -> pd.Series:
    values = values.astype(float)
    rolling = values.rolling(window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return ((values - mean) / std.mask(std.eq(0))).mask(std.eq(0) & mean.notna(), 0.0)


def build_v3_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Calculate V3 from ordered 1m OHLCV, trade count and taker base volume.

    `timestamp` is the candle open time. No target or future row is consulted.
    Warmup NaNs are retained; callers must reject an incomplete final row.
    """
    required = ("timestamp", "open", "high", "low", "close", "volume",
                "trade_count", "taker_buy_base")
    if candles.empty or any(name not in candles for name in required):
        raise ValueError("V3 requires timestamp, OHLCV, trade_count and taker_buy_base")
    frame = candles.loc[:, required].copy().reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("V3 requires unique ordered timestamps")
    if len(frame) > 1 and not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("V3 requires continuous 1m candles")
    numeric = frame.drop(columns="timestamp").astype("float64")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("V3 has unexpected NaN/Inf or invalid prices/activity")
    if (numeric[["volume", "trade_count", "taker_buy_base"]] < 0).any().any() or (numeric.taker_buy_base - numeric.volume > 1e-8).any():
        raise ValueError("V3 received invalid activity")

    close, open_, high, low = (numeric[name] for name in ("close", "open", "high", "low"))
    volume = numeric.volume
    result = pd.DataFrame({"timestamp": frame.timestamp})
    result["return_1m"] = close.pct_change(fill_method=None)
    for horizon in (3, 5, 15):
        result[f"return_{horizon}m"] = close / close.shift(horizon) - 1
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    strength = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + strength)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result["rsi_14"] = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["ema_distance"] = (close - ema20) / close
    result["macd_relative"] = macd / close
    result["macd_signal_relative"] = signal / close
    result["macd_histogram_relative"] = (macd - signal) / close
    result["candle_range"] = (high - low) / close
    result["body_return"] = (close - open_) / open_
    result["upper_wick"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    result["lower_wick"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
    result["rolling_volatility"] = result.return_1m.rolling(window=20, min_periods=20).std(ddof=0)
    result["volume_change"] = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    result["volume_zscore"] = _zscore(volume)
    result["trades_zscore"] = _zscore(numeric.trade_count)
    result["taker_buy_ratio"] = numeric.taker_buy_base / volume.mask(volume.eq(0))
    return result.loc[:, ("timestamp", *V3_FEATURE_COLUMNS)]
