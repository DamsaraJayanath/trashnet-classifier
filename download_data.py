
import shutil
from pathlib import Path
import kagglehub


def download_trashnet(data_path: Path = Path("data")):
    if data_path.exists():
        print(f"{data_path} already exists...")
        return

    print(f"{data_path} does not exist... creating & downloading")

    data_path.mkdir(parents=True, exist_ok=True)

    # Download dataset
    path = kagglehub.dataset_download("feyzazkefe/trashnet")
    print("Downloaded to:", path)

    # Copy dataset
    shutil.copytree(path, data_path, dirs_exist_ok=True)

    print(f"Dataset copied to {data_path}")


if __name__ == "__main__":
    download_trashnet()
