"""Dataset io utils module."""

import glob
import os.path

from evdad.types import PathStr


def parse_dataset_input(path: PathStr, ext: str) -> list:
    if os.path.isdir(path):
        return glob.glob(f"{path}/**/*.{ext}", recursive=True)
    return read_csv(path)


def read_csv(path: PathStr) -> list:
    """Reads data from csv file.

    Args:
        path: path to .csv file

    Returns:
        List of rows from .csv file
    """
    with open(path, mode="r", encoding="utf-8") as file:
        # return [line.rstrip("\n").split(',') for line in file.readlines()]
        return [line.rstrip("\n") for line in file.readlines()]
