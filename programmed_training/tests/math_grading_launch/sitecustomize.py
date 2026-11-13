"""Opt-in bootstrap for the real math recipe's offline module-launch test."""

import os

if os.environ.get("MATH_GRADING_LAUNCH_MARKER"):
    from math_grading_cli_driver import install

    install()