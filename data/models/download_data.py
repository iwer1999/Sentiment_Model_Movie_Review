"""Downloads and extracts the IMDb Large Movie Review Dataset (Maas et al., 2011)."""
import ssl
import tarfile
import certifi
import urllib.request
from pathlib import Path

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if (DATA_DIR / "aclImdb").exists():
        print("Dataset already extracted. Nothing to do.")
        return

    if not ARCHIVE.exists():
        print("Downloading dataset (about 80 MB)...")

        # Create an SSL context using the certifi CA bundle
        context = ssl.create_default_context(cafile=certifi.where())

        # Open the URL with the custom context and save it
        with urllib.request.urlopen(URL, context=context) as response:
            with open(ARCHIVE, 'wb') as out_file:
                out_file.write(response.read())

        print("Download finished.")

    print("Extracting (this can take a few minutes)...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):   # safer extraction on newer Python
            tar.extractall(DATA_DIR, filter="data")
        else:
            tar.extractall(DATA_DIR)
    print("Done. Dataset is in the 'data/aclImdb' folder.")


if __name__ == "__main__":
    main()