"""Compatibility wrapper for the centralized raw-data importer.

Driver standings are generated together with the other raw tables because they
depend on the historical race-results pass.
"""

from generate_raw_data import main


if __name__ == "__main__":
    main()
