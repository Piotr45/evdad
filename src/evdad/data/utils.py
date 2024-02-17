"""Dataset utils module"""


def read_csv(path: str) -> list:
    """Reads data from csv file.

    Args:
        path: path to .csv file

    Returns:
        List of rows from .csv file
    """
    with open(path, mode="r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file.readlines()]
