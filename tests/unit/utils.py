import os.path

from evdad.types import PathStr


def get_test_data_dir() -> PathStr:
    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "test_data"))
