import logging
import zipfile
import kaggle

from pathlib import Path


def download_dataset(owner: str, name: str, dest: Path, logger: logging.Logger):
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.dataset_download_files(dataset=f"{owner}/{name}", path=dest)


def unzip_file(src: Path, name: str, logger: logging.Logger) -> Path:
    zip_name = name + ".zip"
    extract_to = src / name

    if extract_to.is_dir():
        logger.info("[ %s ] is already unzipped. Skipping ...", name)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", name, extract_to)
        with zipfile.ZipFile(src / zip_name, "r") as zip_ref:
            zip_ref.extractall(src / name)

    return extract_to
