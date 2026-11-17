"""Shared command-line options so every entry point has the same --backend / --config-dir flags."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional

from robot.config import Config, load_config
from robot.hal import BACKENDS, Backend, make_backend


def add_common_args(p: argparse.ArgumentParser, default_backend: str = "sim") -> argparse.ArgumentParser:
    p.add_argument("--backend", choices=BACKENDS, default=default_backend,
                   help=f"real robot, MuJoCo sim, or dry (no hardware, logs only). Default {default_backend}")
    p.add_argument("--config-dir", help="config directory (default config/stack)")
    p.add_argument("--viewer", action="store_true",
                   help="sim: open the MuJoCo viewer (on macOS run with `mjpython` instead of `python`)")
    p.add_argument("--fast", action="store_true", help="sim/dry: run faster than realtime")
    p.add_argument("--yes", "-y", action="store_true", help="real: skip the 'support the robot' prompt")
    p.add_argument("--no-imu", action="store_true",
                   help="real: run without the MPU9250 (no fall detection; arm tests only)")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING")
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")


def backend_from_args(args: argparse.Namespace, cfg: Optional[Config] = None,
                      **opts: Any) -> "tuple[Config, Backend]":
    """Load config and create the backend chosen on the command line."""
    cfg = cfg or load_config(args.config_dir)
    extra: Dict[str, Any] = dict(opts)
    if args.backend == "sim":
        extra.setdefault("viewer", bool(getattr(args, "viewer", False)))
    if args.backend == "real" and getattr(args, "no_imu", False):
        extra["imu"] = False
        logging.getLogger(__name__).warning(
            "running WITHOUT the IMU: fall detection is off, the robot always reads as upright")
    if args.backend == "real" and not getattr(args, "yes", False):
        print("REAL ROBOT: every servo will go to the stand pose (offset + 0) and stay stiff.")
        print("Support the robot, keep it still (gyro calibration), then press ENTER. Ctrl-C aborts.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            sys.exit(1)
    return cfg, make_backend(args.backend, cfg, **extra)


def realtime_from_args(args: argparse.Namespace) -> bool:
    """Real robot is always realtime; sim/dry unless --fast."""
    return args.backend == "real" or not getattr(args, "fast", False)
