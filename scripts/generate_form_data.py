"""Compatibility wrapper for the centralized raw-data importer.

Recent-form features are generated together with the other raw tables because
they depend on previous race results for each driver.
"""

from generate_raw_data import main


if __name__ == "__main__":
    main()
