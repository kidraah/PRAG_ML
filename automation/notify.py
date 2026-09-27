"""
PRAG — Desktop Notifications (macOS)

A banner for the moments someone should know about: a login waiting on a
security code, a daily digest that's ready. Failing to notify never fails
a run.
"""

import subprocess
import sys

# The texts go in as arguments, so quotes in them can't break the script.
SCRIPT = (
    "on run argv\n"
    "display notification (item 2 of argv) with title (item 1 of argv) subtitle (item 3 of argv)\n"
    "end run"
)


def notify(title: str, message: str, subtitle: str = "") -> bool:
    """Show a notification; False if it couldn't be shown."""
    if sys.platform != "darwin":
        return False
    try:
        subprocess.run(["osascript", "-e", SCRIPT, title, message, subtitle],
                       capture_output=True, timeout=10, check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
