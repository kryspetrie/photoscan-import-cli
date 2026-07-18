"""Allow running as ``python -m com.kryspetrie.photoscan_import_cli`` (delegates to photocrop CLI)."""
from com.kryspetrie.photoscan_import_cli.photocrop import main

if __name__ == "__main__":
    main()