#!/usr/bin/env python3

import subprocess


def take_screenshot(filepath=""):
    cmd = [
        "gz", "service",
        "-s", "/gui/screenshot",
        "--reqtype", "gz.msgs.StringMsg",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'data: "{filepath}"'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Screenshot failed")


if __name__ == "__main__":
    # nome automatico
    take_screenshot()

    # oppure file custom
    # take_screenshot("/home/user/.gz/gui/pictures/test.png")