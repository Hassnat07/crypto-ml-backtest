"""Download historical spot kline (candle) data from Binance's public data archive.

No API key required. Data is served as monthly zipped CSVs from
https://data.binance.vision — see README for the URL pattern.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_INTERVALS = ["1d", "4h"]
DEFAULT_START = (2020, 1)

CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

INTERVAL_TIMEDELTA = {
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
}

# Values above this are microsecond epochs, below are millisecond epochs.
MICROSECOND_THRESHOLD = 10**14


@dataclass(frozen=True)
class MonthlyFile:
    symbol: str
    interval: str
    year: int
    month: int

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.year:04d}-{self.month:02d}.zip"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.symbol}/{self.interval}/{self.filename}"

    @property
    def dest_path(self) -> Path:
        return RAW_DIR / self.filename


def last_completed_month(today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    last_day_of_prev_month = today.replace(day=1) - timedelta(days=1)
    return last_day_of_prev_month.year, last_day_of_prev_month.month


def month_sequence(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    start_year, start_month = start
    end_year, end_month = end
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def download_file(url: str, dest: Path, retries: int = 3, backoff_seconds: float = 1.0) -> bool:
    """Download url to dest. Returns True if the file exists at dest afterwards.

    A 404 is treated as "no such file" (e.g. symbol didn't exist yet) and returns
    False without raising or retrying. Other failures are retried with backoff.
    """
    if dest.exists():
        return True

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException:
            if attempt == retries - 1:
                return False
            time.sleep(backoff_seconds * (2**attempt))
            continue

        if response.status_code == 404:
            return False
        if response.status_code != 200:
            if attempt == retries - 1:
                return False
            time.sleep(backoff_seconds * (2**attempt))
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return True

    return False


def download_all(symbols: list[str], intervals: list[str], start: tuple[int, int], end: tuple[int, int]) -> list[MonthlyFile]:
    """Download every symbol/interval/month combo, skipping 404s. Returns files that exist locally."""
    months = month_sequence(start, end)
    plan = [
        MonthlyFile(symbol=symbol, interval=interval, year=year, month=month)
        for symbol in symbols
        for interval in intervals
        for year, month in months
    ]

    available: list[MonthlyFile] = []
    for entry in tqdm(plan, desc="Downloading klines", unit="file"):
        if download_file(entry.url, entry.dest_path):
            available.append(entry)

    return available


def _detect_has_header(data: bytes) -> bool:
    first_line = data.split(b"\n", 1)[0]
    first_cell = first_line.split(b",")[0].strip()
    try:
        float(first_cell)
    except ValueError:
        return True
    return False


def parse_kline_zip(zip_path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        data = zf.read(csv_name)

    has_header = _detect_has_header(data)
    df = pl.read_csv(
        io.BytesIO(data),
        has_header=False,
        skip_rows=1 if has_header else 0,
        new_columns=CSV_COLUMNS,
        infer_schema_length=0,
    )
    df = df.drop("ignore")

    open_time_sample = int(df["open_time"][0])
    divisor = 1000 if open_time_sample > MICROSECOND_THRESHOLD else 1

    df = df.with_columns(
        pl.from_epoch(pl.col("open_time").cast(pl.Int64) // divisor, time_unit="ms").alias("open_time"),
        pl.from_epoch(pl.col("close_time").cast(pl.Int64) // divisor, time_unit="ms").alias("close_time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("quote_volume").cast(pl.Float64),
        pl.col("trade_count").cast(pl.Int64),
        pl.col("taker_buy_base").cast(pl.Float64),
        pl.col("taker_buy_quote").cast(pl.Float64),
    )
    return df


def combine_files(files: list[MonthlyFile]) -> pl.DataFrame:
    frames = [parse_kline_zip(f.dest_path) for f in files]
    df = pl.concat(frames)
    df = df.sort("open_time").unique(subset=["open_time"], keep="first").sort("open_time")
    return df


def validate(df: pl.DataFrame, symbol: str, interval: str) -> None:
    row_count = df.height
    first_date = df["open_time"].min()
    last_date = df["open_time"].max()

    step = INTERVAL_TIMEDELTA[interval]
    expected_count = int((last_date - first_date) / step) + 1
    missing_count = max(expected_count - row_count, 0)

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    null_count = int(df.select(pl.col(ohlcv_cols).null_count()).sum_horizontal()[0])

    bad_high_low = df.filter(pl.col("high") < pl.col("low")).height
    bad_high_open = df.filter(pl.col("high") < pl.col("open")).height
    bad_high_close = df.filter(pl.col("high") < pl.col("close")).height

    open_times = df["open_time"].to_list()
    strictly_increasing = all(a < b for a, b in zip(open_times, open_times[1:]))

    print(f"\n{symbol} {interval}")
    print(f"  rows:            {row_count}")
    print(f"  first:           {first_date}")
    print(f"  last:            {last_date}")
    print(f"  expected candles:{expected_count}")
    print(f"  missing/gaps:    {missing_count}")

    assert bad_high_low == 0, f"{symbol} {interval}: {bad_high_low} rows have high < low"
    assert bad_high_open == 0, f"{symbol} {interval}: {bad_high_open} rows have high < open"
    assert bad_high_close == 0, f"{symbol} {interval}: {bad_high_close} rows have high < close"
    assert null_count == 0, f"{symbol} {interval}: {null_count} null values in OHLCV columns"
    assert strictly_increasing, f"{symbol} {interval}: open_time is not strictly increasing"

    print("  validation:      OK")


def process_symbol_interval(symbol: str, interval: str, files: list[MonthlyFile]) -> None:
    matching = [f for f in files if f.symbol == symbol and f.interval == interval]
    if not matching:
        print(f"\n{symbol} {interval}: no files available, skipping")
        return

    df = combine_files(matching)
    validate(df, symbol, interval)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARQUET_DIR / f"{symbol}_{interval}.parquet"
    df.write_parquet(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance historical kline data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START[0])
    parser.add_argument("--start-month", type=int, default=DEFAULT_START[1])
    parser.add_argument("--end-year", type=int, default=None, help="Defaults to last completed month's year")
    parser.add_argument("--end-month", type=int, default=None, help="Defaults to last completed month's month")
    args = parser.parse_args()

    if args.end_year is not None and args.end_month is not None:
        end = (args.end_year, args.end_month)
    else:
        end = last_completed_month()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = download_all(args.symbols, args.intervals, (args.start_year, args.start_month), end)

    for symbol in args.symbols:
        for interval in args.intervals:
            process_symbol_interval(symbol, interval, files)


if __name__ == "__main__":
    main()
