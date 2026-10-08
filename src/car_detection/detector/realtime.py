"""Compatibility entry point for the standalone detector."""

from .realtime_cli import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
