import hydra
import torch


def get_device() -> str:
    """Function that checks which device is avalible.

    Returns:
        Torch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger_path(filename: str) -> str:
    """Gets logger path from hydra."""
    return f"{get_hydra_dir_path()}/{filename}"


def get_hydra_dir_path() -> str:
    """Gets hydra run dir path."""
    return hydra.utils.to_absolute_path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
