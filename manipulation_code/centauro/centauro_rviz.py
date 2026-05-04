#!/usr/bin/env python3

from pathlib import Path
import os
import sys


def main():
    rviz_config = Path(__file__).resolve().with_name("centauro.rviz")

    if not rviz_config.is_file():
        print(f"Configurazione RViz non trovata: {rviz_config}", file=sys.stderr)
        return 1

    os.execvp("rviz2", ["rviz2", "-d", str(rviz_config)])


if __name__ == "__main__":
    raise SystemExit(main())
