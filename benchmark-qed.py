#!/usr/bin/env python3
"""Run benchmark-qed AutoQ without manual PYTHONPATH or virtualenv setup.

Examples:
  python3 benchmark-qed.py
  python3 benchmark-qed.py --config minimal_settings.yaml --output /storage/output
  python3 benchmark-qed.py --generation-types data_local data_global
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> int:
    proc = subprocess.run(cmd, env=env)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def discover_runner() -> tuple[str, str]:
    """Return (runner_type, runner_value) for benchmark-qed.

    runner_type:
      - "binary": invoke via executable, runner_value is binary path
      - "module": invoke via `python -m <module>`, runner_value is module name
    """
    exe = shutil.which("benchmark-qed")
    if exe:
        return ("binary", exe)

    module_candidates = [
        "benchmark_qed",
        "benchmark_qed.cli",
        "benchmark_qed.__main__",
    ]
    for module_name in module_candidates:
        probe = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return ("module", module_name)

    return ("", "")


def install_benchmark_qed() -> None:
    print("benchmark-qed not found. Installing with pip --user...")
    run([sys.executable, "-m", "pip", "install", "--user", "benchmark-qed"], check=True)


def build_env(extra_pythonpaths: list[Path]) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    ordered: list[str] = []

    for p in extra_pythonpaths:
        sp = str(p)
        if sp not in ordered:
            ordered.append(sp)

    if existing:
        for part in existing.split(os.pathsep):
            if part and part not in ordered:
                ordered.append(part)

    env["PYTHONPATH"] = os.pathsep.join(ordered)
    return env


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Local wrapper for benchmark-qed autoq")
    p.add_argument(
        "--config",
        type=str,
        default=str(here / "minimal_settings_fast.yaml"),
        help="Path to benchmark-qed settings YAML",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/storage/output_fast",
        help="Output directory for benchmark-qed artifacts",
    )
    p.add_argument(
        "--generation-types",
        nargs="+",
        default=["data_local"],
        help="One or more generation types (for example: data_local)",
    )
    p.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Do not attempt pip install if benchmark-qed is missing",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and environment only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    config = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not config.exists():
        raise SystemExit(f"Config file not found: {config}")

    output.mkdir(parents=True, exist_ok=True)

    # Include this folder in PYTHONPATH so `local_hf_provider` imports work without manual export.
    env = build_env([here])

    runner_type, runner_value = discover_runner()
    if not runner_type:
        if args.no_auto_install:
            raise SystemExit(
                "benchmark-qed is not installed and --no-auto-install was set. "
                "Install it with: python3 -m pip install --user benchmark-qed"
            )
        install_benchmark_qed()
        runner_type, runner_value = discover_runner()
        if not runner_type:
            raise SystemExit(
                "benchmark-qed still not found after install. "
                "Try: python3 -m pip install --user benchmark-qed"
            )

    if runner_type == "binary":
        cmd = [
            runner_value,
            "autoq",
            str(config),
            str(output),
            "--generation-types",
            *args.generation_types,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            runner_value,
            "autoq",
            str(config),
            str(output),
            "--generation-types",
            *args.generation_types,
        ]

    print("Running:", " ".join(cmd))
    print("PYTHONPATH:", env.get("PYTHONPATH", ""))

    if args.dry_run:
        return

    code = run(cmd, env=env, check=False)
    if code != 0:
        raise SystemExit(code)

    print("Done. Output directory:", output)


if __name__ == "__main__":
    main()
