"""Compatibility wrapper for the centralized raw-data importer.

Circuit features are generated together with the other raw tables because they
share the same Jolpica API calls and race identifiers.
"""

from generate_raw_data import main


if __name__ == "__main__":
    main()
