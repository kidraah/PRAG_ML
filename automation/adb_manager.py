"""
PRAG — ADB Device State Isolator

Wipes Instagram app data and randomises the Android device ID via ADB
so each account cycle looks like a fresh device to Instagram's servers.
"""

import random
import string
import subprocess


def _run_adb(command: str) -> str:
    """Execute an ADB shell command and return stdout."""
    result = subprocess.run(
        ["adb", "shell", command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  ⚠️  ADB command failed: {command}")
        print(f"      stderr: {result.stderr.strip()}")
    return result.stdout.strip()


def wipe_device_state() -> None:
    """Clear Instagram app data and randomise the device's android_id.

    Steps
    -----
    1. ``pm clear com.instagram.android`` — removes all app data
       (login tokens, caches, shared-prefs).
    2. Writes a new random 16-hex-char ``android_id`` into the secure
       settings so the next login appears to come from a different
       device.
    """
    print("\n🧹  Wiping device state …")

    # 1. Clear Instagram data
    output = _run_adb("pm clear com.instagram.android")
    print(f"  ✔  pm clear → {output}")

    # 2. Randomise android_id
    new_id = "".join(random.choices(string.hexdigits[:16], k=16))
    _run_adb(f"settings put secure android_id {new_id}")
    print(f"  ✔  android_id → {new_id}")

    print("  ✔  Device state reset complete.\n")


def get_device_info() -> dict:
    """Return basic device info for logging / debugging."""
    return {
        "android_id": _run_adb("settings get secure android_id"),
        "model": _run_adb("getprop ro.product.model"),
        "sdk": _run_adb("getprop ro.build.version.sdk"),
    }


def force_stop_instagram() -> None:
    """Force-stop the Instagram app via ADB."""
    _run_adb("am force-stop com.instagram.android")
    print("  ✔  Instagram force-stopped.")


def launch_instagram() -> None:
    """Launch Instagram's main activity via ADB."""
    subprocess.run(
        [
            "adb", "shell", "am", "start",
            "-n", "com.instagram.android/com.instagram.android.activity.MainTabActivity",
        ],
        capture_output=True,
        timeout=10,
    )
    print("  ✔  Instagram launched via ADB.")


if __name__ == "__main__":
    print("Device info:", get_device_info())
    wipe_device_state()
    print("New device info:", get_device_info())

