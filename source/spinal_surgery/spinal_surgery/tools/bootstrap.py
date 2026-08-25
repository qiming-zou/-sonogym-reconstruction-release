"""Compatibility wrapper for ``python -m spinal_surgery.tools.bootstrap``."""

from sonogym_bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main())
