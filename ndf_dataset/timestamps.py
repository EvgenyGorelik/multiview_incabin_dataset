"""Parse nanosecond timestamps and find nearest entries in sorted timelines."""

from bisect import bisect_left
from pathlib import Path


def filename_timestamp(filename):
    try:
        return int(Path(filename).stem)
    except ValueError:
        return None


def nearest_timestamp(timestamps, timestamp):
    """Return the nearest timestamp, preferring earlier ties; None if empty."""
    if not timestamps:
        return None
    index = bisect_left(timestamps, timestamp)
    return min(timestamps[max(0, index - 1):index + 1],
               key=lambda ts: (abs(ts - timestamp), ts))
