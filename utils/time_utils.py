"""
Timestamp and time utility functions.
"""

import time
from datetime import datetime, timedelta
from typing import Optional


def get_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp as string."""
    return datetime.fromtimestamp(timestamp).strftime(format_str)


def get_time_window(start_time: float, end_time: float) -> float:
    """Get duration of time window in seconds."""
    return end_time - start_time


def is_within_window(timestamp: float, window_start: float, window_end: float) -> bool:
    """Check if timestamp is within time window."""
    return window_start <= timestamp <= window_end


def get_recent_timestamp(seconds_ago: float) -> float:
    """Get timestamp from N seconds ago."""
    return time.time() - seconds_ago
