"""
Datetime utilities for timezone handling and formatting.

This module provides functions for converting and formatting datetimes
with timezone awareness.
"""

import os
import logging
import pytz

# Module-level logger
logger = logging.getLogger(__name__)


def local_datetime_filter(dt):
    """
    Convert a UTC datetime object to the user's local timezone and return as ISO 8601.

    Returns an ISO 8601 string with timezone offset (e.g. "2026-03-31T01:31:00-04:00")
    so the frontend can parse it unambiguously with new Date().

    Args:
        dt: datetime object to format (assumed UTC if naive)

    Returns:
        str: ISO 8601 datetime string with timezone offset
    """
    if dt is None:
        return ""

    # Get timezone from .env, default to UTC
    user_tz_name = os.environ.get('TIMEZONE', 'UTC')
    try:
        user_tz = pytz.timezone(user_tz_name)
    except pytz.UnknownTimeZoneError:
        user_tz = pytz.utc
        logger.warning(f"Invalid TIMEZONE '{user_tz_name}' in .env. Defaulting to UTC.")

    # If the datetime object is naive, assume it's UTC
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)

    # Convert to the user's timezone
    local_dt = dt.astimezone(user_tz)

    # Return ISO 8601 with timezone offset for unambiguous frontend parsing
    return local_dt.isoformat()
