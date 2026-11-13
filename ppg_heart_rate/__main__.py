"""Allow ``python -m ppg_heart_rate`` to run the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
