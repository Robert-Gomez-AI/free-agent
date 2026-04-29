from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given IANA timezone (e.g. 'UTC', 'America/Bogota').

    Use this whenever the user asks about the current time, today's date, or anything
    time-relative ('how long until X', 'what day is it'). Do not guess.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone!r}. Try an IANA name like 'UTC' or 'America/Bogota'."
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# Authoritative list of tools shipped with the package.
# User tools (created via /tool new) are merged on top by tools.registry at runtime.
BUILTIN_TOOLS = [current_time]
