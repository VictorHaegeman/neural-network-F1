"""Compatibility wrapper for the centralized raw-data importer.

Race-control fields are V0 placeholders/proxies until a richer source is
integrated.
"""

from generate_raw_data import main


if __name__ == "__main__":
    main()
